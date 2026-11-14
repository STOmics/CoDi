import datetime
import logging
import os
import random
import time
from collections import Counter, defaultdict
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from contrastive_augmentation import augment_data
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from utils import plot_embeddings_umap

OUTPUT_DIRS = ["loss_curves", "models"]
PATIENCE = 15  # Early stopping: epochs without validation improvement

timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
logger = logging.getLogger(__name__)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CombinedLoss(nn.Module):
    def __init__(self, temperature=1, contrastive_weight=0.75, class_weights=None):
        """Weighted sum of a contrastive loss (cosine similarity) and cross-entropy.

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
            contrastive_weight (float, optional): weight of the contrastive loss; the
                classification loss gets 1 - contrastive_weight. Defaults to 0.75.
            class_weights (torch.tensor, optional): class weights for cross-entropy.
        """
        super().__init__()
        self.device = get_device()
        self.temperature = torch.tensor(temperature).to(self.device)
        self.contrastive_weight = torch.tensor(contrastive_weight).to(self.device)
        self.class_weights = class_weights
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights)

    def _forward_2(self, z_i, z_j):
        """Loss of anchor vs positive batch; the other 2*(N-1) samples are negatives.

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        z = torch.cat([z_i, z_j], dim=0)
        similarity = cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool))
            .float()
            .to(self.device)
        )
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        return torch.sum(all_losses) / (2 * batch_size)

    def forward(self, z_i, z_j, z_k, anchor_target=None, log_reg=None):
        """Returns (weighted contrastive loss, weighted classification loss).

        The classification loss is 0 when `anchor_target` is None.
        """
        positive_loss = self._forward_2(z_i, z_j)
        if torch.is_tensor(z_k):  # Single (dummy) negative: no negative loss
            negative_loss = 1
        else:
            negative_loss = sum(self._forward_2(z_i, neg) for neg in z_k)
        contrastive_loss = positive_loss / negative_loss

        log_loss = 0 if anchor_target is None else self.cross_entropy(log_reg, anchor_target)
        return (
            contrastive_loss * self.contrastive_weight,
            log_loss * (1 - self.contrastive_weight),
        )


class MLP(torch.nn.Sequential):
    """Multi-layer perceptron with linearly shrinking layers, BatchNorm, PReLU and dropout."""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2.")

        step = (input_dim - hidden_dim) // (n_layers - 1)
        hidden_sizes = [input_dim - i * step for i in range(1, n_layers)]

        layers = []
        in_dim = input_dim
        for hidden_dim_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, hidden_dim_size))
            # Batch norm before activation: https://arxiv.org/abs/1502.03167
            layers.append(torch.nn.BatchNorm1d(hidden_dim_size))
            layers.append(nn.PReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim_size

        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        super().__init__(*layers)


class DeepEncoder(nn.Module):
    def __init__(
        self, input_dim, emb_dim, encoder_depth=4, classifier_depth=2, out_dim=1
    ):
        """Deep encoder: 1. encoder MLP, 2. optional classifier MLP, 3. linear head.

        Args:
            input_dim (int): size of the inputs
            emb_dim (int): dimension of the embedding space
            encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
            classifier_depth (int, optional): number of layers of the classifier MLP. Defaults to 2.
            out_dim (int, optional): number of output classes. Defaults to 1.
        """
        super().__init__()
        self.classifier_depth = classifier_depth
        self.encoder = MLP(input_dim, emb_dim, encoder_depth, dropout=0.2)
        if self.has_classifier:
            self.classifier = MLP(emb_dim, emb_dim, classifier_depth, dropout=0.2)
        self.linear = torch.nn.Linear(emb_dim, out_dim)

        self.encoder.apply(self._init_weights)
        if self.has_classifier:
            self.classifier.apply(self._init_weights)
        self.linear.apply(self._init_weights)

    @property
    def has_classifier(self):
        return self.classifier_depth > 0

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def _classify(self, emb):
        return self.classifier(emb) if self.has_classifier else emb

    def forward(self, anchor, random_pos, random_neg):
        emb_anchor = self.get_logits(anchor)
        emb_positive = self.get_logits(random_pos)

        if random_neg.dim() == 3:  # (batch, n_negatives, features)
            # All negatives go through the encoder first, then the classifier
            emb_negative = [self.encoder(neg) for neg in random_neg.permute(1, 0, 2)]
            if self.has_classifier:
                emb_negative = [self.classifier(emb) for emb in emb_negative]
        else:
            emb_negative = self.get_logits(random_neg)

        log_pred = self.linear(emb_anchor)
        return emb_anchor, emb_positive, emb_negative, log_pred

    def get_logits(self, input_data):
        return self._classify(self.encoder(input_data))

    def get_embeddings(self, input_data):
        return self.encoder(input_data)

    def get_log_reg(self, input_data):
        return torch.softmax(self.linear(self.get_logits(input_data)), dim=1)


class ContrastiveDataset(Dataset):
    def __init__(self, data, target, num_of_negatives, columns=None):
        """Yields (anchor, positive, negatives, anchor target) samples.

        The positive is a random other sample of the anchor's class; negatives
        are random samples from the other classes.
        """
        self.data = np.array(data)
        self.target = np.array(target)
        self.target_to_group = defaultdict(list)
        self.target_to_opposite = defaultdict(list)
        for ut in np.unique(target):
            is_ut = target == ut
            self.target_to_group[ut] = np.flatnonzero(is_ut).tolist()
            self.target_to_opposite[ut] = np.flatnonzero(~is_ut).tolist()
        self.columns = columns
        self.num_of_negatives = num_of_negatives

    def __getitem__(self, index):
        anchor_target = self.target[index]
        anchor_sample = torch.tensor(self.data[index], dtype=torch.float)

        # Positive: random sample from the same class, other than the anchor
        same_group = self.target_to_group[anchor_target]
        random_idx = index
        while random_idx == index:
            random_idx = random.choice(same_group)
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)

        # Negatives: random samples from the other classes
        if self.num_of_negatives > 0:
            opposite_group = self.target_to_opposite[anchor_target]
            random_opposite_idxs = random.sample(opposite_group, k=self.num_of_negatives)
            random_opposite_samples = torch.tensor(
                self.data[random_opposite_idxs], dtype=torch.float
            )
        else:
            # Anchor sent as a dummy negative; the loss ignores it
            random_opposite_samples = anchor_sample

        return anchor_sample, random_sample, random_opposite_samples, anchor_target

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape


def make_dataset(X, y, num_of_negatives):
    """ContrastiveDataset from a numpy array or DataFrame."""
    is_df = isinstance(X, pd.DataFrame)
    return ContrastiveDataset(
        X.to_numpy() if is_df else X,
        y,
        num_of_negatives=num_of_negatives,
        columns=X.columns if is_df else None,
    )


def make_training_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        shuffle=shuffle,
    )


class ContrastiveEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        num_of_negatives=2,
        batch_size=512,
        epochs=50,
        emb_dim=16,
        encoder_depth=4,
        classifier_depth=2,
        contrastive_only_perc=0.3,
        contrastive_weight=0.8,
        freeze_encoder=True,
        out_dim=1,
        freeze_hidden=True,
        verbose=0,
        class_weights=None,
    ):
        """Contrastive learning encoder with logistic regression.

        Minimizes the contrastive loss of a sample and its positive and negative views.

            Args:
                num_of_negatives (int, optional): number of random negative samples picked from
                                                  classes other than the anchor's. Defaults to 2.
                batch_size (int, optional): number of samples in the batch. Defaults to 512.
                epochs (int, optional): number of epochs to train deep encoder. Defaults to 50.
                emb_dim (int, optional): Dimension of the output embeddings. Defaults to 16.
                encoder_depth (int, optional): Number of layers in the encoder MLP. Defaults to 4.
                classifier_depth (int, optional): Number of layers in the classifier MLP. Defaults to 2.
                contrastive_only_perc (float [0, 1], optional): % of epochs to use only contrastive loss for training
                contrastive_weight (float [0, 1], optional): Weight of contrastive loss when combined with the
                                                             cross-entropy loss of logistic regression:
                                                             L_total = cw * L_contrastive + (1-cw) * L_classification
                freeze_encoder (Boolean, optional): Unused.
                freeze_hidden (Boolean, optional): Freeze weights of the classifier MLP during the first
                                                   contrastive_only_perc of epochs.
                out_dim (int, optional): Dimension of the output result. Defaults to 1.
        """
        self.num_of_negatives = num_of_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.emb_dim = emb_dim
        self.encoder_depth = encoder_depth
        self.classifier_depth = classifier_depth
        self.out_dim = out_dim
        self.contrastive_only_perc = contrastive_only_perc
        self.contrastive_weight = contrastive_weight
        self.freeze_encoder = freeze_encoder
        self.freeze_hidden = freeze_hidden
        self.verbose = verbose
        self.class_weights = class_weights
        self.model = None
        self.device = get_device()
        self.train_loss_history_contrastive = []
        self.train_loss_history_classification = []
        self.val_loss_history_contrastive = []
        self.val_loss_history_classification = []
        self.sum_loss = []

    def _new_model(self):
        return DeepEncoder(
            input_dim=self.contr_ds.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            encoder_depth=self.encoder_depth,
            classifier_depth=self.classifier_depth,
        )

    def _is_contrastive_only(self, epoch):
        return epoch <= self.contrastive_only_perc * self.epochs

    def _run_epoch(self, loader, combined_loss, contrastive_only, optimizer=None, progress=None):
        """One pass over `loader`; trains when an optimizer is given, else validates.

        Returns:
            (mean weighted contrastive loss, mean weighted classification loss)
        """
        training = optimizer is not None
        self.model.train(training)
        sum_contrastive = 0.0
        sum_classification = 0.0

        with torch.set_grad_enabled(training):
            for anchor, positive, negative, anchor_target in loader:
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                anchor_target = anchor_target.to(self.device)

                if training:
                    optimizer.zero_grad()

                emb_anchor, emb_positive, emb_negative, log_reg = self.model(
                    anchor, positive, negative
                )

                if training and self.freeze_hidden and self.classifier_depth > 0:
                    # Classifier is frozen while only the contrastive loss is used
                    for param in self.model.classifier.parameters():
                        param.requires_grad = not contrastive_only

                if contrastive_only:
                    loss_contrastive, loss_classification = combined_loss(
                        emb_anchor, emb_positive, emb_negative
                    )
                else:
                    loss_contrastive, loss_classification = combined_loss(
                        emb_anchor, emb_positive, emb_negative, anchor_target, log_reg
                    )

                if training:
                    loss = loss_contrastive + loss_classification
                    loss.backward()
                    optimizer.step()

                sum_contrastive += anchor.size(0) * loss_contrastive.item()
                sum_classification += anchor.size(0) * loss_classification.item()

                if progress is not None:
                    progress.set_postfix({"loss": loss.item()})

        num_samples = len(loader.dataset)
        return sum_contrastive / num_samples, sum_classification / num_samples

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Train a DeepEncoder that pulls an anchor towards its positive view (same class)
        and pushes it away from its negative views (other classes).

        Args:
            X (Pandas Dataframe): Rows are samples, columns are features
            y (Array): Targets for all samples in data
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.data = X_train
        self.y = y_train
        # Cannot sample more negatives than the smallest class has samples
        self.num_of_negatives = np.min(
            [self.num_of_negatives] + list(Counter(y).values())
        )

        self.contr_ds = make_dataset(X_train, y_train, self.num_of_negatives)
        self.contr_ds_val = make_dataset(X_val, y_val, self.num_of_negatives)
        train_loader = make_training_loader(self.contr_ds, self.batch_size, shuffle=True)
        val_loader = make_training_loader(self.contr_ds_val, self.batch_size)

        self.model = self._new_model().to(self.device)
        optimizer = Adam(
            [
                {"params": self.model.encoder.parameters(), "lr": 0.01},
                {"params": self.model.classifier.parameters(), "lr": 0.001},
            ]
        )
        combined_loss = CombinedLoss(
            class_weights=self.class_weights, temperature=0.5
        ).to(self.device)

        self.train_loss_history_contrastive = []
        self.train_loss_history_classification = []
        self.val_loss_history_contrastive = []
        self.val_loss_history_classification = []
        self.sum_loss = []

        best_val_loss = float("inf")
        no_improvement_count = 0

        epochs = range(1, self.epochs + 1)
        progress = tqdm(epochs, leave=False) if self.verbose > 0 else None
        for epoch in progress if progress is not None else epochs:
            start = time.time()
            contrastive_only = self._is_contrastive_only(epoch)

            train_contrastive, train_classification = self._run_epoch(
                train_loader, combined_loss, contrastive_only, optimizer, progress
            )
            val_contrastive, val_classification = self._run_epoch(
                val_loader, combined_loss, contrastive_only
            )

            self.train_loss_history_contrastive.append(train_contrastive)
            self.train_loss_history_classification.append(train_classification)
            self.sum_loss.append(train_contrastive + train_classification)
            self.val_loss_history_contrastive.append(val_contrastive)
            self.val_loss_history_classification.append(val_classification)

            logger.info(f"Epoch {epoch} took {time.time() - start:.2f}s")

            # Early stopping only once the classification loss is included
            if not contrastive_only:
                avg_val_loss = val_contrastive + val_classification
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= PATIENCE:
                    logger.info(
                        f"Early stopping after {epoch+1} epochs without improvement."
                    )
                    break

        return self

    def load_model(self, path: str):
        """Create a DeepEncoder and load its parameters from a .pt file."""
        self.model = self._new_model()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def _transform_or_predict(self, get_emb_func, X=None, y=None):
        """Apply `get_emb_func` of the trained model to X (or to the training data)."""
        if X is None:
            loader = DataLoader(
                self.contr_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            )
        else:
            ds = make_dataset(X, np.ones(len(X)), num_of_negatives=0)
            loader = DataLoader(ds, batch_size=self.batch_size, pin_memory=True)

        self.model.eval()
        with torch.no_grad():
            outputs = [get_emb_func(anchor.to(self.device)) for anchor, _, _, _ in loader]
        return torch.cat(outputs)

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        preds = self._transform_or_predict(self.model.get_log_reg, X, y)
        return torch.argmax(preds, dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return self._transform_or_predict(self.model.get_log_reg, X, y).cpu().numpy()

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return self._transform_or_predict(self.model.get_embeddings, X, y)

    def get_embeddings_true(self, X: np.ndarray):
        return self._transform_or_predict(self.model.get_embeddings, X)

    def set_params(self, **params):
        """Set known attributes; unknown keys are ignored."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def plot_loss_curve(ce, path):
    fig, axs = plt.subplots(2)
    axs[0].plot(ce.train_loss_history_contrastive, label="Training")
    axs[0].plot(ce.val_loss_history_contrastive, label="Validation")
    axs[1].plot(ce.train_loss_history_classification, label="Training")
    axs[1].plot(ce.val_loss_history_classification, label="Validation")

    plt.title("Model loss")
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[0].set_ylabel("contrastive loss")
    axs[0].set_xlabel("epoch")
    axs[1].set_ylabel("classification loss")
    axs[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(path)

    fig, ax = plt.subplots()
    ax.plot(ce.sum_loss, color="black")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.savefig(f"loss_curves/total_loss_size_{timestamp}.png")

    logger.info(f"Saved the loss curves .png to {path}")


def log_embeddings_and_augmented_data(ce, X, y, adata_sc, adata_st, sc_path, st_path):
    """Save UMAP plots and embeddings of SC and ST data, and the augmented SC data."""
    plot_embeddings_umap(
        ce,
        X,
        y,
        figname=os.path.basename(sc_path).replace(".h5ad", "_SC_UMAP.png"),
        save_embeddings=True,
    )
    basename, ext = os.path.splitext(os.path.basename(sc_path))
    adata_sc.write_h5ad(f"{basename}_AUGMENTED{ext}")

    plot_embeddings_umap(
        ce,
        adata_st.X.toarray(),
        adata_st.obs["contrastive"],
        figname=os.path.basename(st_path).replace(".h5ad", "_ST_UMAP.png"),
        save_embeddings=True,
    )


def contrastive_process(
    sc_path: str,
    st_path: str,
    adata_sc: ad.AnnData,
    adata_st: ad.AnnData,
    annotation_sc: str,
    batch_size: int,
    epochs: int,
    embedding_dim: int,
    encoder_depth: int,
    classifier_depth: int,
    filename: str,
    augmentation_perc: float,
    logger: logging.Logger,
    log_embeddings: bool,
    queue=None,
):
    """Train the contrastive model on (augmented) SC data and predict ST cell types.

    Puts the ST class probabilities and then the ST predictions on `queue`.
    """
    for d in OUTPUT_DIRS:
        os.makedirs(d, exist_ok=True)

    fix_seed(0)
    adata_sc.X = adata_sc.layers["counts"]
    adata_sc = augment_data(
        adata_sc,
        adata_st,
        annotation=annotation_sc,
        percentage=augmentation_perc,
        logger=logger,
    )

    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.normalize_total(adata_st, target_sum=1e4)

    X = adata_sc.X.toarray()
    logger.info("Input ready...")

    y = adata_sc.obs[annotation_sc]
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    logger.info("Labels ready...")

    ce = ContrastiveEncoder(
        out_dim=len(le.classes_),
        batch_size=batch_size,
        epochs=epochs,
        emb_dim=embedding_dim,
        encoder_depth=encoder_depth,
        classifier_depth=classifier_depth,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_le, test_size=0.1, random_state=42, stratify=y_le
    )
    logger.info(
        f"""Fitting a model with: \n - embedding dim: {embedding_dim} \n - encoder depth: {encoder_depth} \n - classifier depth: {classifier_depth} \n - epochs: {epochs}"""
    )
    ce.fit(X_train, y_train)
    logger.info("Finished training...")
    model_save_path = f"models/{X.shape[0]}_cells_{timestamp}.pt"
    ce.save_model(model_save_path)
    logger.info(f"Saved the model to {model_save_path}")
    plot_loss_curve(
        ce,
        f"loss_curves/loss_size_{X.shape[0]}_cells_{len(le.classes_)}_epochs_{epochs}_{timestamp}.png",
    )

    y_true = le.inverse_transform(y_test)
    y_pred = le.inverse_transform(ce.predict(X_test))
    logger.info("-------------Test SC data------------")
    logger.info(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    logger.info(f"F1 macro score: {f1_score(y_true, y_pred, average='macro')}")
    logger.info("----------------------------------")

    logger.info("-------------ST prediction------------")
    adata_st.var_names_make_unique()
    if not scipy.sparse.issparse(adata_st.X):
        adata_st.X = scipy.sparse.csr_matrix(adata_st.X)
        logger.info("Converted gene exp matrix of ST to csr_matrix")
    # Single forward pass: the predicted class is the argmax of the probabilities
    probabilities = ce.predict_proba(adata_st.X.toarray())
    adata_st.obs["contrastive"] = le.inverse_transform(np.argmax(probabilities, axis=1))
    adata_st.obs.index.name = "cell_id"

    if log_embeddings:
        log_embeddings_and_augmented_data(ce, X, y, adata_sc, adata_st, sc_path, st_path)

    df_probabilities = pd.DataFrame(
        data=probabilities, columns=le.classes_, index=adata_st.obs.index
    )
    if queue:
        queue.put(df_probabilities)
        queue.put(adata_st.obs["contrastive"])

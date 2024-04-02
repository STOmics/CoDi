import argparse as ap
from collections import Counter
from collections import defaultdict
import datetime
import logging
from typing import Optional
import time
import random
import os
import sys


import anndata as ad
from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score

from core.contrastive_augmentation import augment_data
from core.preprocessing import preprocess

dirs = ["loss_curves", "models"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    def __init__(self, temperature=1, contrastive_weight=0.75, class_weights=None):
        """Loss for contrastive learning using cosine distance as similarity metric.

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = torch.tensor(temperature).to(self.device)
        self.contrastive_weight = torch.tensor(contrastive_weight).to(self.device)
        self.class_weights = class_weights

    def _forward_2(self, z_i, z_j):
        """Compute loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # First, one needs to apply an L2 normalization to the features
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # compute similarity between the sample's embeddings
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
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

    def forward(self, z_i, z_j, z_k, anchor_target=None, log_reg=None):
        positive_loss = self._forward_2(z_i, z_j)
        negative_loss = 0
        if torch.is_tensor(z_k):
            negative_loss = 1
        else:
            for neg in z_k:
                negative_loss += self._forward_2(z_i, neg)
        contrastive_loss = positive_loss / negative_loss

        if anchor_target is None:
            log_loss = 0
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            log_loss = criterion(log_reg, anchor_target)
        log_reg_loss = contrastive_loss * self.contrastive_weight + log_loss * (
            1 - self.contrastive_weight
        )
        # return a tuple representing both contrastive and classification loss
        return (
            contrastive_loss * self.contrastive_weight,
            log_loss * (1 - self.contrastive_weight),
        )


class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2.")

        step = (input_dim - hidden_dim) // (n_layers - 1)
        hidden_sizes = [input_dim - i * step for i in range(1, n_layers)]

        for hidden_dim_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, hidden_dim_size))
            # added batch norm before activation func.: https://arxiv.org/abs/1502.03167
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
        """Implementation of deep encoder consisted of: 1. Input layer MLP, 2. Hidden layer MLP 3. Linear.
        Args:
            input_dim (int): size of the inputs
            emb_dim (int): dimension of the embedding space
            encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
            classifier_depth (int, optional): number of layers of the hidden head. Defaults to 2.
        """
        super().__init__()
        self.classifier_depth = classifier_depth
        self.encoder = MLP(input_dim, emb_dim, encoder_depth, dropout=0.2)
        if self.classifier_depth > 0:
            self.classifier = MLP(emb_dim, emb_dim, classifier_depth, dropout=0.2)
        self.linear = torch.nn.Linear(emb_dim, out_dim)

        # initialize weights
        self.encoder.apply(self._init_weights)
        if self.classifier_depth > 0:
            self.classifier.apply(self._init_weights)
        self.linear.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_pos, random_neg):
        positive = random_pos
        negative = random_neg.permute(1, 0, 2) if random_neg.dim() == 3 else random_neg

        # compute embeddings
        emb_anchor = self.encoder(anchor)
        if self.classifier_depth > 0:
            emb_anchor = self.classifier(emb_anchor)

        emb_positive = self.encoder(positive)
        if self.classifier_depth > 0:
            emb_positive = self.classifier(emb_positive)

        emb_negative = (
            [self.encoder(neg_sample) for neg_sample in negative]
            if random_neg.dim() == 3
            else self.encoder(negative)
        )
        if self.classifier_depth > 0:
            emb_negative = (
                [self.classifier(emb_neg) for emb_neg in emb_negative]
                if random_neg.dim() == 3
                else self.classifier(emb_negative)
            )

        log_pred = self.linear(emb_anchor)

        return emb_anchor, emb_positive, emb_negative, log_pred

    def get_embeddings(self, input_data):
        emb_anchor = self.encoder(input_data)
        if self.classifier_depth > 0:
            emb_anchor = self.classifier(emb_anchor)
        return emb_anchor

    def get_log_reg(self, input_data):
        emb_anchor = self.get_embeddings(input_data)
        log_prediction = torch.softmax(self.linear(emb_anchor), dim=1)
        return log_prediction


class ContrastiveDataset(Dataset):
    def __init__(self, data, target, num_of_negatives, columns=None):
        self.data = np.array(data)
        self.target = np.array(target)
        unique_targets = np.unique(target)
        self.target_to_group = defaultdict(list)
        for ut in unique_targets:
            self.target_to_group[ut] = [
                ind for ind, element in enumerate((target == ut)) if element
            ]
        self.target_to_opposite = defaultdict(list)
        for ut in unique_targets:
            self.target_to_opposite[ut] = [
                ind for ind, element in enumerate((target != ut)) if element
            ]
        self.columns = columns
        self.num_of_negatives = num_of_negatives

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        anchor_target = self.target[index]
        anchor_sample = torch.tensor(self.data[index], dtype=torch.float)

        # Fetch positive (random from same endpoint (target) group)
        same_group = self.target_to_group[anchor_target]

        random_idx = index
        while random_idx == index:  # avoid comparing same samples
            random_idx = random.choice(same_group)
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)

        # Fetch negative (random from oposite endpoint (target) group
        opposite_group = self.target_to_opposite[anchor_target]

        if self.num_of_negatives > 0:
            random_oposite_idxs = random.sample(opposite_group, k=self.num_of_negatives)
            # random_oposite_idx = [opposite_group[x] for x in oposite_group_indices]
            random_oposite_samples = torch.tensor(
                self.data[random_oposite_idxs], dtype=torch.float
            )
        else:
            # Send anchor sample: Will be ignored by calculating always zero loss
            random_oposite_samples = anchor_sample

        return anchor_sample, random_sample, random_oposite_samples, anchor_target

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape


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
        """Implementation of Contrastive learning encoder with logistic regression.
        It is done by minimizing the contrastive loss of a sample and its positive and negative view.

            Args:
                num_of_negatives (int, optional): number of random negative samples picket randomply from endpoint group
                                                  oposite to anchor.Defaults to 2.
                batch_size (int, optional): number of samples in the batch. Defaults to 512.
                epochs (int, optional): number of to train deep encoder. Defaults to 50.
                emb_dim (int, optional): Dimension of the output embeddings. Defaults to 16.
                encoder_depth (int, optional): Number of layers in the input MLP layer. Defaults to 4.
                classifier_depth (int, optional): Number of layers in the classifier MLP layer. Defaults to 2.
                contrastive_only_perc (float [0, 1], optional): % of epochs to use only contrastive loss for training
                contrastive_weight (float [0, 1], optional): Weight of contrastive loss  when combined in cross-entropy
                                                             loss of logistic regression with formula:
                                                             L_total = cw * L_contrastive + (1-cw) * L_classification
                freeze_encoder (Boolean, optional): Freeze weights of input layer (Multy-Layer Perceptron) after
                                                    contrastive_only_perc of epochs.
                freeze_hidden (Boolean, optional): Freeze weights of hidden layer (Multy-Layer Perceptron) after
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loss_history_contrastive = []
        self.train_loss_history_classification = []
        self.val_loss_history_contrastive = []
        self.val_loss_history_classification = []
        self.sum_loss = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Instantiate and train DeepEncoder that will try to minimize the loss between anchor and positive view
        (random sample from same endpoint group like anchor) and maximize the loss between anchor and negative view(s).

        Args:
            X (Pandas Dataframe): Rows are samples, columns are features
            y (Array): Binary targets for all samples in data
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.data = X_train
        self.y = y_train
        # Limit number of negatives
        self.num_of_negatives = np.min(
            [self.num_of_negatives] + list(Counter(y).values())
        )

        self.contr_ds = ContrastiveDataset(
            X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train,
            y_train,
            num_of_negatives=self.num_of_negatives,
            columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None,
        )

        self.contr_ds_val = ContrastiveDataset(
            X_val.to_numpy() if isinstance(X_val, pd.DataFrame) else X_val,
            y_val,
            num_of_negatives=self.num_of_negatives,
            columns=X_val.columns if isinstance(X_val, pd.DataFrame) else None,
        )

        train_loader = DataLoader(
            self.contr_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
            shuffle=True,
        )

        val_loader = DataLoader(
            self.contr_ds_val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=3,
        )

        self.model = DeepEncoder(
            input_dim=self.contr_ds.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            encoder_depth=self.encoder_depth,
            classifier_depth=self.classifier_depth,
        ).to(self.device)

        params = [
            {"params": self.model.encoder.parameters(), "lr": 0.01},
            {"params": self.model.classifier.parameters(), "lr": 0.001},
        ]

        optimizer = Adam(params)
        # optimizer = SGD(params, weight_decay=0.1)
        combined_loss = CombinedLoss(
            class_weights=self.class_weights, temperature=0.5
        ).to(self.device)

        train_loss_history_contrastive = []
        train_loss_history_classification = []
        val_loss_history_contrastive = []
        val_loss_history_classification = []
        sum_loss = []

        best_val_loss = float("inf")
        no_improvement_count = 0
        PATIENCE = 15

        # scaler = GradScaler()

        # Train contrastive learning encoder
        epochs_tqdm = (
            tqdm(range(1, self.epochs + 1), leave=False)
            if self.verbose > 0
            else range(1, self.epochs + 1)
        )
        for epoch in epochs_tqdm:
            start = time.time()
            self.model.train()
            train_loss_contrastive = 0.0
            train_loss_classification = 0.0

            for anchor, positive, negative, anchor_target in train_loader:
                anchor, positive, negative, anchor_target = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    negative.to(self.device),
                    anchor_target.to(self.device),
                )

                # reset gradients
                optimizer.zero_grad()

                # get embeddings
                emb_anchor, emb_positive, emb_negative, log_reg = self.model(
                    anchor, positive, negative
                )

                # compute loss
                if epoch <= self.contrastive_only_perc * self.epochs:
                    # consider only contrastive loss until encoder trains enough
                    # freeze classifier weights for training only on contrastive loss
                    if self.freeze_hidden and self.classifier_depth > 0:
                        for param in self.model.classifier.parameters():
                            param.requires_grad = False
                    loss_contrastive, loss_classification = combined_loss(
                        emb_anchor, emb_positive, emb_negative
                    )
                else:
                    #  Take logistic reg loss into account
                    # if self.freeze_encoder:
                    #     # First, freeze weights of contrastive encoder!
                    #     for param in self.model.encoder.parameters():
                    #         param.requires_grad = False
                    # unfreeze weights of the classifier to use combined loss
                    if self.freeze_hidden and self.classifier_depth > 0:
                        for param in self.model.classifier.parameters():
                            param.requires_grad = True
                    loss_contrastive, loss_classification = combined_loss(
                        emb_anchor,
                        emb_positive,
                        emb_negative,
                        anchor_target,
                        log_reg,
                    )
                loss = loss_contrastive + loss_classification
                loss.backward()

                # update model weights
                optimizer.step()

                # log progress
                train_loss_contrastive += anchor.size(0) * loss_contrastive.item()
                train_loss_classification += anchor.size(0) * loss_classification.item()

                if self.verbose > 0:
                    epochs_tqdm.set_postfix({"loss": loss.item()})

            # validation
            self.model.eval()
            val_loss_contrastive = 0.0
            val_loss_classification = 0.0
            with torch.no_grad():
                for anchor, positive, negative, anchor_target in val_loader:
                    anchor, positive, negative, anchor_target = (
                        anchor.to(self.device),
                        positive.to(self.device),
                        negative.to(self.device),
                        anchor_target.to(self.device),
                    )
                    # get embeddings
                    emb_anchor, emb_positive, emb_negative, log_reg = self.model(
                        anchor, positive, negative
                    )

                    # compute loss
                    if epoch <= self.contrastive_only_perc * self.epochs:
                        # consider only contrastive loss until encoder trains enough
                        contrastive_loss, classification_loss = combined_loss(
                            emb_anchor, emb_positive, emb_negative
                        )
                    else:
                        #  Take logistic reg loss into account
                        contrastive_loss, classification_loss = combined_loss(
                            emb_anchor,
                            emb_positive,
                            emb_negative,
                            anchor_target,
                            log_reg,
                        )
                    val_loss_contrastive += anchor.size(0) * contrastive_loss.item()
                    val_loss_classification += (
                        anchor.size(0) * classification_loss.item()
                    )

            avg_train_loss_contrastive = train_loss_contrastive / len(
                train_loader.dataset
            )
            avg_train_loss_classification = train_loss_classification / len(
                train_loader.dataset
            )
            train_loss_history_contrastive.append(avg_train_loss_contrastive)
            train_loss_history_classification.append(avg_train_loss_classification)

            sum_loss.append(avg_train_loss_contrastive + avg_train_loss_classification)

            avg_val_loss_contrastive = val_loss_contrastive / len(val_loader.dataset)
            avg_val_loss_classification = val_loss_classification / len(
                val_loader.dataset
            )
            val_loss_history_contrastive.append(avg_val_loss_contrastive)
            val_loss_history_classification.append(avg_val_loss_classification)

            end = time.time()
            elapsed = end - start
            logger.info(f"Epoch {epoch} took {elapsed:.2f}s")

            avg_val_loss = avg_val_loss_contrastive + avg_val_loss_classification
            is_classification_loss_included = (
                epoch > self.contrastive_only_perc * self.epochs
            )
            if is_classification_loss_included:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= PATIENCE:
                    # pass
                    logger.info(
                        f"Early stopping after {epoch+1} epochs without improvement."
                    )
                    break

        self.train_loss_history_contrastive = train_loss_history_contrastive
        self.train_loss_history_classification = train_loss_history_classification
        self.val_loss_history_contrastive = val_loss_history_contrastive
        self.val_loss_history_classification = val_loss_history_classification
        self.sum_loss = sum_loss
        return self

    def load_model(self, path: str):
        """Creates DeepEncoder and loads it from path

        Args:
            path (str): path to the .pt file that has model params
        """
        self.model = DeepEncoder(
            input_dim=self.contr_ds.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            encoder_depth=self.encoder_depth,
            classifier_depth=self.classifier_depth,
        )
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def _transform_or_predict(self, get_emb_func, X=None, y=None):
        """Perform encoding on previously trained Contrastive encoder. If data is not provided it will process the
        same data used for fit()
        """
        if X is None:
            loader = DataLoader(
                self.contr_ds,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
            )
        else:
            ds = ContrastiveDataset(
                X.to_numpy() if isinstance(X, pd.DataFrame) else X,
                np.ones(len(X)),
                num_of_negatives=0,
                columns=X.columns if isinstance(X, pd.DataFrame) else None,
            )
            loader = DataLoader(ds, batch_size=self.batch_size, pin_memory=True)

        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for anchor, _, _, _ in loader:
                anchor = anchor.to(self.device)
                embeddings.append(get_emb_func(anchor))
        embeddings = torch.cat(embeddings)
        return embeddings

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        preds = self._transform_or_predict(self.model.get_log_reg, X, y)
        preds = torch.argmax(preds, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        preds = self._transform_or_predict(self.model.get_log_reg, X, y)
        return preds.cpu().numpy()

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        return self._transform_or_predict(self.model.get_embeddings, X, y)

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                pass

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
    queue=None,
):
    if filename:
        file_handler = logging.FileHandler(filename)
        logger.addHandler(file_handler)

    fix_seed(0)
    adata_sc.X = adata_sc.layers["counts"]

    # preprocess(adata_sc)
    adata_sc = augment_data(
        adata_sc, adata_st, annotation=annotation_sc, percentage=augmentation_perc, logger=logger
    )

    # perform preprocessing like removing all 0 vectors, normalization and scaling

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

    y_pred = ce.predict(X_test)

    y_true = y_test
    acc = accuracy_score(le.inverse_transform(y_true), le.inverse_transform(y_pred))
    f1 = f1_score(
        le.inverse_transform(y_true), le.inverse_transform(y_pred), average="macro"
    )

    logger.info("-------------Test data------------")
    logger.info(f"Accuracy: {acc}")
    logger.info(f"F1 macro score: {f1}")
    logger.info("----------------------------------")

    logger.info("-------------ST prediction------------")
    adata_st.var_names_make_unique()
    if not scipy.sparse.issparse(adata_st.X):
        adata_st.X = scipy.sparse.csr_matrix(adata_st.X)
        logger.info(f"Converted gene exp matrix of ST to csr_matrix")
    y_pred = ce.predict(adata_st.X.toarray())
    adata_st.obs["contrastive"] = le.inverse_transform(y_pred)
    adata_st.obs.index.name = "cell_id"
    probabilities = ce.predict_proba(adata_st.X.toarray())
    df_probabilities = pd.DataFrame(
        data=probabilities, columns=le.classes_, index=adata_st.obs.index
    )
    if queue:
        queue.put(df_probabilities)
        queue.put(adata_st.obs["contrastive"])

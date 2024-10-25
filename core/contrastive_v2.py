import argparse as ap
from collections import Counter
from collections import defaultdict
import datetime
import logging
from typing import Optional
import time
import random
import socket
import os
from pathlib import Path
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
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

from contrastive_augmentation import augment_data
from utils import plot_embeddings_umap
from preprocessing import preprocess
from losses import SupConLoss
import wandb

dirs = ["loss_curves", "models"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)


def init_wandb(model_folder="weights", name="", model=None):
    wandb_output_dir = os.path.join(model_folder, "wandb_home")
    Path(wandb_output_dir).mkdir(parents=True, exist_ok=True)
    wandb.require("core")
    run = wandb.init(
        project="contrastive_lr",
        notes=socket.gethostname(),
        name=f"{name}_contrastive_lr_{timestamp}",
        group="classifier",
        dir=wandb_output_dir,
        job_type="training",
        reinit=True,
    )
    if model:
        wandb.watch(model, log="all")
    return run


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.min_validation_loss = float("inf")


timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
logger = logging.getLogger(__name__)


class SupervisedContrastiveLoss(nn.Module):
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

    def _forward_2(self, z_i, z_j, labels):
        """Compute loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # First, one needs to apply an L2 normalization to the features
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        contrast_feature = torch.cat([z_i, z_j], dim=0)
        anchor_count = 2
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # all strategy
        dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(dot_contrast, dim=1, keepdim=True)
        logits = dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, anchor_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Apply class weights using the labels as indices
        if self.class_weights is not None:
            weights = self.class_weights[labels.squeeze().repeat(anchor_count)].to(device)  # get weights for each label in the batch
            mean_log_prob_pos *= weights.squeeze()  # apply weights to mean log probability

        # loss
        loss = -(self.temperature / self.temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def forward(self, z_i, z_j, anchor_target=None):
        positive_loss = self._forward_2(z_i, z_j, anchor_target)
        return positive_loss

        if anchor_target is None or log_reg is None:
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

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.2):
        layers = []
        in_dim = input_dim
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2.")

        step = (input_dim - hidden_dim) // (n_layers - 1)
        hidden_sizes = [input_dim - i * step for i in range(1, n_layers)]
        logger.info(f"Hidden sizes: {hidden_sizes}")

        for hidden_dim_size in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, hidden_dim_size))
            # added batch norm before activation func.: https://arxiv.org/abs/1502.03167
            layers.append(torch.nn.BatchNorm1d(hidden_dim_size))
            layers.append(nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim_size
        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        super().__init__(*layers)


class DeepEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        head_type="linear",
        encoder_depth=4,
        classifier_depth=2,
        out_dim=1,
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
        self.encoder = MLP(input_dim, emb_dim, encoder_depth, dropout=0.1)
        if head_type == "linear":
            self.classifier = nn.Linear(emb_dim, out_dim)
        elif head_type == "mlp":
            self.classifier = MLP(emb_dim, out_dim, classifier_depth, dropout=0.1)
        else:
            raise NotImplementedError(f"Not supported head type: {head_type}")
        self.linear = torch.nn.Identity()

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.classifier.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_pos, classify=False):
        positive = random_pos

        # compute embeddings
        emb_anchor = self.encoder(anchor)
        if classify:
            emb_anchor = self.classifier(emb_anchor)

        emb_positive = self.encoder(positive)
        if classify:
            emb_positive = self.classifier(emb_positive)

        return emb_anchor, emb_positive

    def get_embeddings(self, input_data):
        emb = self.encoder(input_data)
        emb = self.classifier(emb)
        return emb

    def get_embeddings_true(self, input_data):
        emb = self.encoder(input_data)
        return emb

    def get_log_reg(self, input_data):
        emb_anchor = self.get_embeddings(input_data)
        log_prediction = torch.softmax(self.linear(emb_anchor), dim=1)
        return log_prediction

    def freeze_encoder_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


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
        emb_dim=32,
        encoder_depth=4,
        classifier_depth=2,
        contrastive_only_perc=0.5,
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
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = torch.tensor(class_weights).float().to(self.device)

    def train_encoder(self):
        pass

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
            head_type="linear",
            encoder_depth=self.encoder_depth,
            classifier_depth=self.classifier_depth,
        ).to(self.device)

        optimizer = Adam(self.model.encoder.parameters(), lr=3e-4, weight_decay=1e-2)

        sup_con_loss = SupervisedContrastiveLoss(
            class_weights=self.class_weights, temperature=0.5
        ).to(self.device)

        ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=3, factor=0.5
        )

        early_stopper = EarlyStopper(patience=10)

        # Training loop encoder
        for epoch in range(self.epochs):

            self.model.train()
            train_loss_contrastive = 0.0

            for anchor, positive, _, anchor_target in train_loader:
                anchor, positive, anchor_target = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    anchor_target.to(self.device),
                )

                # reset gradients
                optimizer.zero_grad()

                # get embeddings
                emb_anchor, emb_positive = self.model(anchor, positive)

                loss = sup_con_loss(emb_anchor, emb_positive, anchor_target)
                loss.backward()

                optimizer.step()

                train_loss_contrastive += loss.item()

            train_loss_contrastive = train_loss_contrastive / len(train_loader)
            wandb.log({"Encoder training loss": train_loss_contrastive})
            scheduler.step(train_loss_contrastive)

            # Validation
            self.model.eval()
            val_loss_contrastive = 0.0
            with torch.no_grad():
                for anchor, positive, _, anchor_target in val_loader:
                    anchor, positive, anchor_target = (
                        anchor.to(self.device),
                        positive.to(self.device),
                        anchor_target.to(self.device),
                    )
                    # get embeddings
                    emb_anchor, emb_positive = self.model(anchor, positive)

                    loss = sup_con_loss(emb_anchor, emb_positive, anchor_target)

                    val_loss_contrastive += loss.item()

            val_loss_contrastive = val_loss_contrastive / len(val_loader)
            wandb.log({"Encoder validation loss": train_loss_contrastive})

            if early_stopper.early_stop(val_loss_contrastive):
                print(f"Early stopping encoder training in epoch {epoch}...")
                break
            else:
                print(
                    f"Finished encoder epoch: {epoch} lr: {optimizer.param_groups[0]['lr']}"
                )

        # Training loop classifier
        # setup optimizer and loss function
        self.model.freeze_encoder_weights()
        early_stopper.reset()
        optimizer = Adam(self.model.classifier.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=3, factor=0.5
        )
        for epoch in range(self.epochs):

            self.model.train()
            train_loss_classification = 0.0

            for anchor, positive, _, anchor_target in train_loader:
                anchor, positive, anchor_target = (
                    anchor.to(self.device),
                    positive.to(self.device),
                    anchor_target.to(self.device),
                )

                # reset gradients
                optimizer.zero_grad()

                # get embeddings
                logits_anchor, logits_positive = self.model(
                    anchor, positive, classify=True
                )

                loss = ce_loss(logits_anchor, anchor_target)
                loss.backward()

                optimizer.step()

                train_loss_classification += loss.item()

            train_loss_classification = train_loss_classification / len(train_loader)
            wandb.log({"Classifier training loss": train_loss_classification})

            # Validation
            self.model.eval()
            val_loss_classification = 0.0
            with torch.no_grad():
                for anchor, positive, _, anchor_target in val_loader:
                    anchor, positive, anchor_target = (
                        anchor.to(self.device),
                        positive.to(self.device),
                        anchor_target.to(self.device),
                    )
                    # get embeddings
                    logits_anchor, logits_positive = self.model(
                        anchor, positive, classify=True
                    )

                    loss = ce_loss(logits_anchor, anchor_target)

                    val_loss_classification += loss.item()

            val_loss_classification = val_loss_classification / len(val_loader)
            wandb.log({"Classifier validation loss": val_loss_classification})
            scheduler.step(val_loss_classification)

            if early_stopper.early_stop(val_loss_classification):
                print(f"Early stopping classifier training in epoch {epoch}...")
                break
            else:
                print(
                    f"Finished classifier epoch: {epoch} lr: {optimizer.param_groups[0]['lr']}"
                )

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

    def get_embeddings(self, X: np.ndarray):
        embs = self._transform_or_predict(self.model.get_embeddings, X)
        return embs

    def get_embeddings_true(self, X: np.ndarray):
        embs = self._transform_or_predict(self.model.get_embeddings_true, X)
        return embs

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


# def plot_loss_curve(ce, path):
#     fig, axs = plt.subplots(2)
#     axs[0].plot(ce.train_loss_history_contrastive, label="Training")
#     axs[0].plot(ce.val_loss_history_contrastive, label="Validation")
#     axs[1].plot(ce.train_loss_history_classification, label="Training")
#     axs[1].plot(ce.val_loss_history_classification, label="Validation")

#     plt.title("Model loss")
#     axs[0].legend(loc="upper left")
#     axs[1].legend(loc="upper left")
#     axs[0].set_ylabel("contrastive loss")
#     axs[0].set_xlabel("epoch")
#     axs[1].set_ylabel("classification loss")
#     axs[1].set_xlabel("epoch")
#     plt.tight_layout()
#     plt.savefig(path)

#     fig, ax = plt.subplots()
#     ax.plot(ce.sum_loss, color="black")
#     ax.set_ylabel("loss")
#     ax.set_xlabel("epoch")
#     ax.set_frame_on(False)
#     plt.tight_layout()
#     plt.savefig(f"loss_curves/total_loss_size_{timestamp}.png")

#     logger.info(f"Saved the loss curves .png to {path}")


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
    queue=None,
    wandb_key: str = "66bdf7f04d7842bb591556f5263dd9c779ca1ce7",
):
    fix_seed(0)
    wandb.login(key=wandb_key)
    run = init_wandb(name=os.path.basename(st_path).replace(".h5ad", ""))
    adata_sc.X = adata_sc.layers["counts"]

    # preprocess(adata_sc)
    # adata_sc = augment_data(
    #     adata_sc,
    #     adata_st,
    #     annotation=annotation_sc,
    #     percentage=augmentation_perc,
    #     logger=logger,
    # )

    # perform preprocessing like removing all 0 vectors, normalization and scaling

    X = adata_sc.X.toarray()
    # sc.pp.normalize_total(adata_sc, target_sum=1e4)
    # sc.pp.log1p(adata_sc)
    # sc.pp.normalize_total(adata_st, target_sum=1e4)
    # sc.pp.log1p(adata_st)
    logger.info("Input ready...")

    y = adata_sc.obs[annotation_sc]
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    logger.info("Labels ready...")

    # compute class weights
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_le), y=y_le
    )

    ce = ContrastiveEncoder(
        out_dim=len(le.classes_),
        batch_size=batch_size,
        epochs=epochs,
        emb_dim=embedding_dim,
        encoder_depth=encoder_depth,
        classifier_depth=classifier_depth,
        class_weights=class_weights,
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
    # plot_loss_curve(
    #     ce,
    #     f"loss_curves/loss_size_{X.shape[0]}_cells_{len(le.classes_)}_epochs_{epochs}_{timestamp}.png",
    # )

    y_pred = ce.predict(X_test)

    y_true = y_test
    acc = accuracy_score(le.inverse_transform(y_true), le.inverse_transform(y_pred))
    f1 = f1_score(
        le.inverse_transform(y_true), le.inverse_transform(y_pred), average="macro"
    )

    plot_embeddings_umap(
        ce,
        X_test,
        le.inverse_transform(y_test),
        os.path.basename(sc_path).replace(".h5ad", "_tsne.png"),
    )

    logger.info("-------------Test data------------")
    logger.info(f"Accuracy: {acc}")
    logger.info(f"F1 macro score: {f1}")
    logger.info("----------------------------------")

    plot_embeddings_umap(
        ce,
        adata_st.X.toarray(),
        figname=os.path.basename(st_path).replace(".h5ad", "_ST_tsne.png"),
    )
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

    wandb.unwatch()
    run.finish()

    if queue:
        queue.put(df_probabilities)
        queue.put(adata_st.obs["contrastive"])

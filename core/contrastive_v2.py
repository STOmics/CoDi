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
import random


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
from ignite.metrics import MaximumMeanDiscrepancy

from contrastive_augmentation import augment_data, augment_st_data
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
    def __init__(self, temperature=0.5, contrastive_weight=0.75, class_weights=None):
        """Loss for contrastive learning using cosine distance as similarity metric.

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = torch.tensor(temperature).to(self.device)
        self.contrastive_weight = torch.tensor(contrastive_weight).to(self.device)
        self.class_weights = class_weights

    def _forward_2(self, z_i, z_j, labels=None):
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
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
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
            weights = self.class_weights[labels.squeeze().repeat(anchor_count)].to(
                device
            )  # get weights for each label in the batch
            mean_log_prob_pos *= (
                weights.squeeze()
            )  # apply weights to mean log probability

        # loss
        loss = -(self.temperature / self.temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def forward(self, z_i, z_j, anchor_target=None):
        positive_loss = self._forward_2(z_i, z_j, anchor_target)
        return positive_loss


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
            # layers.append(torch.nn.Dropout(dropout))
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
        self.encoder = MLP(input_dim, emb_dim, encoder_depth, dropout=0.3)
        if head_type == "linear":
            self.classifier = nn.Linear(emb_dim, out_dim)
        elif head_type == "mlp":
            self.classifier = MLP(emb_dim, out_dim, classifier_depth, dropout=0.3)
        else:
            raise NotImplementedError(f"Not supported head type: {head_type}")

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

    def get_logits(self, input_data):
        emb = self.encoder(input_data)
        emb = self.classifier(emb)
        return emb

    def get_embeddings(self, input_data):
        emb = self.encoder(input_data)
        return emb

    def get_log_reg(self, input_data):
        logits = self.get_logits(input_data)
        predictions = torch.softmax(logits, dim=1)
        return predictions

    def freeze_encoder_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


class STDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        ).to(self.device)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
                :, None, None
            ]
        ).sum(dim=0)


class CosineSimilarityKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, X):
        # Normalize X to ensure cosine similarity is valid
        X_normalized = F.normalize(X, p=2, dim=1)
        # Compute cosine similarity by taking dot products of normalized vectors
        cosine_similarity_matrix = torch.mm(X_normalized, X_normalized.T)
        return cosine_similarity_matrix


class MMDLoss(nn.Module):

    def __init__(self, kernel=CosineSimilarityKernel()):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kernel = kernel.to(self.device)

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


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
        st_data=None,
        supervised_loss=False,
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
        self.st_data = st_data
        self.supervised_loss = supervised_loss

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

        st_ds = STDataset(data=self.st_data)

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

        st_loader = DataLoader(st_ds, batch_size=self.batch_size // 4, shuffle=True)

        self.model = DeepEncoder(
            input_dim=self.contr_ds.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            head_type="mlp",
            encoder_depth=self.encoder_depth,
            classifier_depth=self.classifier_depth,
        ).to(self.device)

        optimizer = Adam(self.model.encoder.parameters(), lr=3e-4, weight_decay=1e-2)

        if self.supervised_loss:
            sup_con_loss = SupervisedContrastiveLoss(
                class_weights=self.class_weights, temperature=0.5
            ).to(self.device)
        else:
            sup_con_loss = SupervisedContrastiveLoss(temperature=0.5).to(self.device)

        mmd_loss = MMDLoss().to(self.device)

        ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=3, factor=0.5
        )

        early_stopper = EarlyStopper(patience=5)

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

                # Get a batch from the ST dataset for MMD calculation
                try:
                    st_batch = next(iter(st_loader))  # Get a batch from ST DataLoader
                except StopIteration:
                    # If you run out of ST data, you may want to restart the ST DataLoader
                    st_loader = DataLoader(
                        st_ds, batch_size=self.batch_size // 4, shuffle=True
                    )
                    st_batch = next(iter(st_loader))

                st_batch = st_batch.to(self.device)
                # st_anchor, st_positive = torch.split(st_batch, self.batch_size)
                st_emb, _ = self.model(st_batch, st_batch)

                # reset gradients
                optimizer.zero_grad()

                # get embeddings
                emb_anchor, emb_positive = self.model(anchor, positive)

                mmd = mmd_loss(torch.cat([emb_anchor, emb_positive]), st_emb)
                if self.supervised_loss:
                    loss = (
                        sup_con_loss(emb_anchor, emb_positive, anchor_target)
                        + 0.1 * mmd
                    )
                else:
                    loss = sup_con_loss(emb_anchor, emb_positive) + 0.1 * mmd
                if random.randint(0, 10) == 3:
                    wandb.log({"MMD loss": mmd})
                # loss = sup_con_loss(emb_anchor, emb_positive, anchor_target)
                loss.backward()

                optimizer.step()

                train_loss_contrastive += loss.item()

            train_loss_contrastive = train_loss_contrastive / len(train_loader)
            wandb.log({"Encoder training loss": train_loss_contrastive})

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
            wandb.log({"Encoder validation loss": val_loss_contrastive})
            scheduler.step(val_loss_contrastive)

            if early_stopper.early_stop(val_loss_contrastive):
                print(f"Early stopping encoder training in epoch {epoch}...")
                break
            else:
                print(
                    f"Finished encoder epoch: {epoch} lr: {optimizer.param_groups[0]['lr']}"
                )

        optimizer = Adam(self.model.encoder.parameters(), lr=1e-4, weight_decay=1e-2)
        con_loss = SupervisedContrastiveLoss(temperature=0.5).to(self.device)
        steps_pretraining_sc = self.epochs * len(train_loader)
        steps_finetuning_st = steps_pretraining_sc * 0.1
        epochs_finetuning_st = int(steps_finetuning_st / len(st_loader))
        print(
            f"Finetuning on ST for steps: {steps_finetuning_st} epochs: {epochs_finetuning_st}"
        )
        # Finetuning on ST data
        for epoch in range(epochs_finetuning_st):
            self.model.train()
            finetune_st_loss = 0.0

            for st_batch in st_loader:

                st_batch_positive = augment_st_data(st_batch)
                st_batch = st_batch.to(self.device)
                st_batch_positive = st_batch_positive.to(self.device)

                st_emb, st_emb_positive = self.model(st_batch, st_batch_positive)

                # reset gradients
                optimizer.zero_grad()

                # mmd = mmd_loss(torch.cat([emb_anchor, emb_positive]), st_emb)
                loss = con_loss(st_emb, st_emb_positive)
                # loss = sup_con_loss(emb_anchor, emb_positive, anchor_target)
                loss.backward()

                optimizer.step()

                finetune_st_loss += loss.item()

            print(f"Finished finetuning encoder epoch: {epoch}")
            finetune_st_loss = finetune_st_loss / len(st_loader)
            wandb.log({"Encoder ST finetuning loss": finetune_st_loss})

        # Training loop classifier
        # setup optimizer and loss function
        self.model.freeze_encoder_weights()
        early_stopper.reset()
        optimizer = Adam(self.model.classifier.parameters(), lr=3e-3, weight_decay=1e-2)
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
        embs = self._transform_or_predict(self.model.get_embeddings, X)
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
    wandb_key: str = "YOUR_WANDB_KEY",
    supervised_loss: bool = False,
):
    fix_seed(0)
    wandb.login(key=wandb_key)
    run = init_wandb(name=os.path.basename(st_path).replace(".h5ad", ""))
    adata_sc.X = adata_sc.layers["counts"]

    # preprocess(adata_sc)
    adata_sc = augment_data(
        adata_sc,
        adata_st,
        annotation=annotation_sc,
        percentage=augmentation_perc,
        logger=logger,
    )

    # perform preprocessing like removing all 0 vectors, normalization and scaling

    sc.pp.normalize_total(adata_sc, target_sum=1e2)
    # sc.pp.log1p(adata_sc)
    sc.pp.normalize_total(adata_st, target_sum=1e2)
    # sc.pp.log1p(adata_st)
    X = adata_sc.X.toarray()
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
        st_data=adata_st.X.toarray(),
        supervised_loss=supervised_loss,
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

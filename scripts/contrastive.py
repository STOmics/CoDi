import argparse as ap
from collections import Counter
from collections import defaultdict
import datetime
import logging
from typing import Optional
import random
import os

from matplotlib import pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

dirs = ["logs", "loss_curves", "contrastive_res", "models"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)

timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    level=logging.INFO,
    filename=f"logs/{timestamp}.log",
)
logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    def __init__(self, temperature=1.0, contrastive_weight=0.75, class_weights=None):
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
        return log_reg_loss


class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))
        super().__init__(*layers)


class DeepEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, encoder_depth=4, hidden_depth=2, out_dim=1):
        """Implementation of deep encoder consisted of: 1. Input layer MLP, 2. Hidden layer MLP 3. Linear.
        Args:
            input_dim (int): size of the inputs
            emb_dim (int): dimension of the embedding space
            encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
            hidden_depth (int, optional): number of layers of the hidden head. Defaults to 2.
        """
        super().__init__()
        self.hidden_depth = hidden_depth
        self.encoder = MLP(input_dim, emb_dim, encoder_depth)
        if self.hidden_depth > 0:
            self.hidden_layer = MLP(emb_dim, emb_dim, hidden_depth)
        self.linear = torch.nn.Linear(emb_dim, out_dim)

        # initialize weights
        self.encoder.apply(self._init_weights)
        if self.hidden_depth > 0:
            self.hidden_layer.apply(self._init_weights)
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
        if self.hidden_depth > 0:
            emb_anchor = self.hidden_layer(emb_anchor)

        emb_positive = self.encoder(positive)
        if self.hidden_depth > 0:
            emb_positive = self.hidden_layer(emb_positive)

        emb_negative = (
            [self.encoder(neg_sample) for neg_sample in negative]
            if random_neg.dim() == 3
            else self.encoder(negative)
        )
        if self.hidden_depth > 0:
            emb_negative = (
                [self.hidden_layer(emb_neg) for emb_neg in emb_negative]
                if random_neg.dim() == 3
                else self.hidden_layer(emb_negative)
            )

        if emb_anchor.dim() == 1:
            log_pred = torch.flatten(torch.sigmoid(self.linear(emb_anchor)))
        else:
            log_pred = self.linear(emb_anchor)

        return emb_anchor, emb_positive, emb_negative, log_pred

    def get_embeddings(self, input_data):
        emb_anchor = self.encoder(input_data)
        if self.hidden_depth > 0:
            emb_anchor = self.hidden_layer(emb_anchor)
        return emb_anchor

    def get_log_reg(self, input_data):
        emb_anchor = self.get_embeddings(input_data)
        if emb_anchor.dim() == 1:
            log_prediction = torch.sigmoid(self.linear(emb_anchor))
        else:
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
            random_oposite_idx = random.sample(opposite_group, k=self.num_of_negatives)
            # random_oposite_idx = [opposite_group[x] for x in oposite_group_indices]
            random_oposite_samples = torch.tensor(
                self.data[random_oposite_idx], dtype=torch.float
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
        batch_size=1024,
        epochs=50,
        emb_dim=16,
        encoder_depth=4,
        hidden_depth=2,
        contrastive_only_perc=0.2,
        contrastive_weight=0.25,
        freeze_encoder=True,
        out_dim=1,
        freeze_hidden=False,
        verbose=0,
        class_weights=None,
    ):
        """Implementation of Contrastive learning encoder with logistic regression.
        It is done by minimizing the contrastive loss of a sample and its positive and negative view.

            Args:
                num_of_negatives (int, optional): number of random negative samples picket randomply from endpoint group
                                                  oposite to anchor.Defaults to 2.
                batch_size (int, optional): number of samples in the batch. Defaults to 16.
                epochs (int, optional): number of to train deep encoder. Defaults to 500.
                emb_dim (int, optional): Dimension of the output embeddings. Defaults to 16.
                encoder_depth (int, optional): Number of layers in the input MLP layer
                hidden_depth (int, optional): Number of layers in the hidden MLP layer
                contrastive_only_perc (float [0, 1], optional): % of epochs to use only contrastive loss for training
                contrastive_weight (float [0, 1], optional): Weight of contrastive loss  when combined in cross-entropy
                                                             loss of logistic regression with formula:
                                                             L_total = cw * L_contrastive + (1-cw) * L_classification
                freeze_encoder (Boolean, optional): Freeze weights of input layer (Multy-Layer Perceptron) after
                                                    contrastive_only_perc of epochs.
                freeze_hidden (Boolean, optional): Freeze weights of hidden layer (Multy-Layer Perceptron) after
                                                   contrastive_only_perc of epochs.
        """
        self.num_of_negatives = num_of_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.emb_dim = emb_dim
        self.encoder_depth = encoder_depth
        self.hidden_depth = hidden_depth
        self.out_dim = out_dim
        self.contrastive_only_perc = contrastive_only_perc
        self.contrastive_weight = contrastive_weight
        self.freeze_encoder = freeze_encoder
        self.freeze_hidden = freeze_hidden
        self.verbose = verbose
        self.class_weights = class_weights
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_history = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Instantiate and train DeepEncoder that will try to minimize the loss between anchor and positive view
        (random sample from same endpoint group like anchor) and maximize the loss between anchor and negative view(s).

        Args:
            X (Pandas Dataframe): Rows are samples, columns are features
            y (Array): Binary targets for all samples in data
        """
        self.data = X
        self.y = y
        # Limit number of negatives
        self.num_of_negatives = np.min(
            [self.num_of_negatives] + list(Counter(y).values())
        )
        self.contr_ds = ContrastiveDataset(
            X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            y,
            num_of_negatives=self.num_of_negatives,
            columns=X.columns if isinstance(X, pd.DataFrame) else None,
        )

        train_loader = DataLoader(
            self.contr_ds,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=3,
        )

        self.model = DeepEncoder(
            input_dim=self.contr_ds.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            encoder_depth=self.encoder_depth,
            hidden_depth=self.hidden_depth,
        ).to(self.device)
        optimizer = Adam(self.model.parameters(), lr=0.001)
        ntxent_loss = CombinedLoss(class_weights=self.class_weights).to(self.device)

        loss_history = []

        # Train contrastive learning encoder
        epochs_tqdm = (
            tqdm(range(1, self.epochs + 1), leave=False)
            if self.verbose > 0
            else range(1, self.epochs + 1)
        )
        for epoch in epochs_tqdm:
            self.model.train()
            epoch_loss = 0.0
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
                if epoch <= self.contrastive_only_perc * self.epochs:  # <= change!!
                    # consider only contrastive loss until encoder trains enough
                    loss = ntxent_loss(emb_anchor, emb_positive, emb_negative)
                else:
                    #  Take logistic reg loss into account
                    if self.freeze_encoder:
                        # First, freeze weights of contrastive encoder!
                        for param in self.model.encoder.parameters():
                            param.requires_grad = False
                    if self.freeze_hidden and self.hidden_depth > 0:
                        for param in self.model.hidden_layer.parameters():
                            param.requires_grad = False
                    loss = ntxent_loss(
                        emb_anchor, emb_positive, emb_negative, anchor_target, log_reg
                    )
                loss.backward()

                # update model weights
                optimizer.step()

                # log progress
                epoch_loss += anchor.size(0) * loss.item()
                if self.verbose > 0:
                    epochs_tqdm.set_postfix({"loss": loss.item()})
            print(f"Epoch: {epoch}")
            epoch_loss = epoch_loss / len(train_loader.dataset)
            loss_history.append(epoch_loss)
        self.loss_history = loss_history
        return self

    def load_model(path: str):
        """Creates DeepEncoder and loads it from path

        Args:
            path (str): path to the .pt file that has model params
        """
        self.model = DeepEncoder(
            input_dim=self.contr_ds.shape[1],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
            encoder_depth=self.encoder_depth,
            hidden_depth=self.hidden_depth,
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
        preds = np.array(
            [x[0] if isinstance(x, (list, tuple, np.ndarray)) else x for x in preds]
        )
        preds_0 = np.array([1 - x for x in preds])
        preds = np.vstack([preds_0, preds]).T
        return preds

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


def subset_marker_genes(adata_sc, adata_st):
    """Calculates marker genes

    Get marker genes on SC dataset by taking the first 100 markers for each cell type and then
    create intersection of those grouped genes with genes in ST dataset.

    Args:
        adata_sc (_type_): _description_
        adata_st (_type_): _description_

    Returns:
        _type_: _description_
    """
    adata_sc.layers["counts"] = adata_sc.X.copy()
    # Calculate marker genes
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    adata_sc.var_names_make_unique()
    sc.tl.rank_genes_groups(adata_sc, groupby=args.annotation, use_raw=False)

    markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
    markers = list(np.unique(markers_df.melt().value.values))

    markers_intersect = list(set(markers).intersection(adata_st.var.index))

    adata_sc = adata_sc[:, markers_intersect]
    adata_st = adata_st[:, markers_intersect]

    return adata_sc, adata_st, markers_intersect


def plot_loss_curve(ce, path):
    plt.plot(ce.loss_history)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig(path)
    logger.info(f"Saved the loss curve .png to {path}")


if __name__ == "__main__":
    parser = ap.ArgumentParser(
        description="A script that trains a contrastive learning model and performs prediction on ST dataset."
    )
    parser.add_argument(
        "--sc_path",
        help="A single cell reference dataset used for training",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--st_path",
        help="A spatially resolved dataset",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs for training",
        type=int,
        default=50,
        required=False,
    )
    parser.add_argument(
        "-a",
        "--annotation",
        help="Annotation label for cell types of single cell dataset",
        type=str,
        required=False,
        default="cell_subclass",
    )
    parser.add_argument(
        "--annotation_st",
        help="Annotation label for cell types of spatially resolved dataset used for testing only",
        type=str,
        required=False,
        default="annotation",
    )
    parser.add_argument(
        "--to_predict",
        help="Datasets to predict on",
        type=str,
        required=False,
        default="",
    )

    args = parser.parse_args()

    adata_sc = sc.read(args.sc_path)
    adata_st = sc.read(args.st_path)
    logger.info("Loaded data...")

    if not scipy.sparse.issparse(adata_sc.X):
        adata_sc.X = scipy.sparse.csr_matrix(adata_sc.X)
        logger.info("Converted SC gene exp matrix to csr")

    if not scipy.sparse.issparse(adata_st.X):
        adata_st.X = scipy.sparse.csr_matrix(adata_st.X)
        logger.info("Converted ST gene exp matrix to csr")

    adata_sc, adata_st, markers_intersect = subset_marker_genes(adata_sc, adata_st)
    logger.info("Found intersecting marker genes, performed subset on ST and SC")

    X = adata_sc.layers["counts"].toarray()
    logger.info("Input ready...")

    y = adata_sc.obs[args.annotation]
    le = LabelEncoder()
    y_le = le.fit_transform(y)
    logger.info("Labels ready...")

    fix_seed(0)
    ce = ContrastiveEncoder(
        out_dim=len(le.classes_),
        epochs=args.epochs,
        emb_dim=64,
        encoder_depth=8,
        hidden_depth=4,
    )
    ce.fit(X, y_le)
    logger.info("Finished training...")
    model_save_path = f"models/{X.shape[0]}_cells_{timestamp}.pt"
    ce.save_model(model_save_path)
    logger.info(f"Saved the model to {model_save_path}")
    plot_loss_curve(
        ce, f"loss_curves/loss_size_{X.shape[0]}_cells_{len(le.classes_)}.png"
    )

    # testing
    logger.info("--------------------------")
    logger.info(f"Model trained on: {args.sc_path}")
    # files = """/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.05.h5ad,/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.1.h5ad,/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.2.h5ad,/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.3.h5ad,/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.5.h5ad,/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.7.h5ad,/goofys/projects/SSI/datasets/4K/Mouse_brain_SC_0.9.h5ad"""
    files = """/goofys/Samples/Stereo_seq/Mouse_brain/SS200000141TL_B5.h5ad"""
    files = files.split(",")
    for file in files:
        ad = sc.read(file)
        ad.var_names_make_unique()
        ad = ad[:, markers_intersect]
        if not scipy.sparse.issparse(ad.X):
            ad.X = scipy.sparse.csr_matrix(ad.X)
            # logger.info(f"Converted SC gene exp matrix of {file} to csr")
        y_pred = ce.predict(ad.X.toarray())
        y_true = ad.obs[args.annotation_st]
        acc = accuracy_score(y_true, le.inverse_transform(y_pred))
        f1 = f1_score(y_true, le.inverse_transform(y_pred), average="macro")
        logger.info(f"Results for: {file}")
        logger.info(f"Accuracy: {acc}")
        logger.info(f"F1 macro score: {f1}")
        logger.info("-------------------------")
        ad.obs["contrastive"] = le.inverse_transform(y_pred)
        fname = file.split(".")[-1]
        ad.obs["contrastive"].to_csv(f"contrastive_res/{fname}.csv")
        # implement saving

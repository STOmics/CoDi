"""
@File    :   losses.py
@Time    :   2024/08/27 13:41:04
@Author  :   Nikola Milicevic
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.02, base_temperature=0.02):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """Compute loss for model.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].

        Returns:
            A loss scalar.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, emb_dim],"
                "3 dimensions are required"
            )

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        mask = torch.eq(labels, labels.T).float().to(device)

        # a number of views for anchor
        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability to avoid large exponentiations
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
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

        mask_pos_pairs = mask.sum(1)
        # This line handles an edge case where some anchors might not have any positive pairs.
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

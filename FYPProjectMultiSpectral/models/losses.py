import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import ModuleConfig

class CombinedFocalLossWithPosWeight(nn.Module):
    def __init__(self, pos_weight, alpha=ModuleConfig.focal_alpha, gamma=ModuleConfig.focal_gamma, reduction='mean'):
        super(CombinedFocalLossWithPosWeight, self).__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute standard BCE loss without reduction
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        probas = torch.sigmoid(inputs) # Convert logits to probabilities
        p_t = probas * targets + (1 - probas) * (1 - targets) # Compute p_t: probability for the true class
        focal_modulation = (1 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets) # Apply alpha balancing: higher weight for positives
        loss = alpha_factor * focal_modulation * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

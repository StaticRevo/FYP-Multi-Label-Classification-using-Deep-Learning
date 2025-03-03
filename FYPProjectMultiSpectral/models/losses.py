import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import ModuleConfig

# --Combined Focal Loss with Pos Weight--
class CombinedFocalLossWithPosWeight(nn.Module):
    def __init__(self, pos_weight, alpha=ModuleConfig.focal_alpha, gamma=ModuleConfig.focal_gamma, reduction='mean'):
        super(CombinedFocalLossWithPosWeight, self).__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none') # Compute standard BCE loss without reduction
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

# --Asymmetric Loss--
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps

    def forward(self, inputs, targets):
        probas = torch.sigmoid(inputs)
        loss_pos = (1 - probas) ** self.gamma_pos * F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        loss_neg = probas ** self.gamma_neg * F.binary_cross_entropy_with_logits(inputs, 1 - targets, reduction='none')
        loss = loss_pos * targets + loss_neg * (1 - targets)
        return loss.mean()

# --Soft F1 Loss--
class SoftF1Loss(nn.Module):
    def __init__(self, smooth=1e-7):
        super(SoftF1Loss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        tp = (probs * targets).sum(dim=0)
        fp = ((1 - targets) * probs).sum(dim=0)
        fn = (targets * (1 - probs)).sum(dim=0)

        f1 = 2 * tp / (2 * tp + fn + fp + self.smooth)
        return 1 - f1.mean()
    
# --Hybrid BCE F1 Loss--a
class HybridBCEF1Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridBCEF1Loss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.f1_loss = SoftF1Loss()

    def forward(self, inputs, targets):
        return self.alpha * self.bce(inputs, targets) + (1 - self.alpha) * self.f1_loss(inputs, targets)


#self.criterion = CombinedFocalLossWithPosWeight(self.class_weights, alpha=ModuleConfig.focal_alpha, gamma=ModuleConfig.focal_gamma, reduction='mean')
#self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, eps=1e-8)
#self.criterion = SoftF1Loss(smooth=1e-7)
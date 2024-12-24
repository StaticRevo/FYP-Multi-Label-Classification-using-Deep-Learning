from torchmetrics import Metric
import torch

class SubsetAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target should be binary tensors
        correct = (preds == target).all(dim=1).sum()
        self.correct += correct
        self.total += target.size(0)
    
    def compute(self):
        return self.correct.float() / self.total

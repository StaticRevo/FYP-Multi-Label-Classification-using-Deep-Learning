import torch
from torchmetrics import Metric

class OneError(Metric):

    def __init__(self, num_labels: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("one_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_labels = num_labels

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Find the index of the top prediction for each sample
        top_idx = preds.argmax(dim=1)
        # Check if the top prediction is correct for each sample
        correct = target[torch.arange(target.size(0)), top_idx] == 1
        errors = (~correct).float()
        self.one_error_sum += errors.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.one_error_sum / self.total if self.total > 0 else torch.tensor(0.0)

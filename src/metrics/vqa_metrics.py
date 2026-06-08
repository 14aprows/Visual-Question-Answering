from pandas._libs.parsers import na_values
import torch 

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def top_k_accuracy(logits, targets, k=5):
    _, top_k_preds = logits.topk(k, dim=1)

    targets = targets.view(-1, 1)
    correct = top_k_preds.eq(targets).any(dim=1).sum().item()
    total = targets.size(0)

    return correct / total
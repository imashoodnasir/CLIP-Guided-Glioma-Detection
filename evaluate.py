
import torch
from sklearn.metrics import confusion_matrix

def compute_metrics(pred, target):
    pred = torch.argmax(pred, dim=1)
    cm = confusion_matrix(target.flatten(), pred.flatten())
    return cm

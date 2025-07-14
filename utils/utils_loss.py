import torch.nn.functional as F
import torchmetrics as tcm
import torch
import matplotlib.pyplot as plt
import numpy as np


class Metrics:
    def __init__(self, num_class, device=None):
        # default to same device as available tensors
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.accuracy  = tcm.Accuracy(task='multiclass', num_classes=num_class).to(self.device)
        self.recall    = tcm.Recall(task='multiclass', num_classes=num_class, average='macro').to(self.device)
        self.precision = tcm.Precision(task='multiclass', num_classes=num_class, average='macro').to(self.device)
        self.f1_score  = tcm.F1Score(task='multiclass', num_classes=num_class, average='macro').to(self.device)

    def compute_metrics(self, preds, target):
        # ensure inputs live on same device as the metrics
        preds   = preds.to(self.device)
        target  = target.to(self.device)

        return [
            self.accuracy(preds, target).item(),
            self.recall(preds, target).item(),
            self.precision(preds, target).item(),
            self.f1_score(preds, target).item()
        ]


def label_cross_entropy(preds, labels):
    return F.binary_cross_entropy_with_logits(preds, labels)


def label_accuracy(preds, target):
    _, preds = preds.max(-1)
    _, target = target.max(-1)
    correct = preds.int().data.eq(target.int()).sum()
    return correct.float() / (target.size(0)), preds[-1], target[-1]


def show_confusion(conf_mat, class_names, show=True):

    fig = plt.figure()

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    if show:
        plt.show()
    else:
        return fig










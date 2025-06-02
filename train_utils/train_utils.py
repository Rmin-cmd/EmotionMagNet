from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.hermitian import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import time
from utils.utils_loss import *
from Model_magnet.encoding_loss_function import *
from sklearn.metrics import confusion_matrix
from scipy.signal import hilbert
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def dataloader(data_train, label_train, A_pdc_train, data_val, label_val,  A_pdc_valid, q, K, batch_size):

    # feature_imag_train = np.imag(hilbert(data_train, axis=-1))
    # feature_imag_valid = np.imag(hilbert(data_val, axis=-1))

    feature_real_train, feature_imag_train = torch.FloatTensor(data_train.reshape(-1, data_train.shape[-1])).to(device),\
                                             torch.FloatTensor(data_train.reshape(-1, data_train.shape[-1])).to(device)

    feature_real_valid, feature_imag_valid = torch.FloatTensor(data_val.reshape(-1, data_val.shape[-1])).to(device),\
                                             torch.FloatTensor(data_val.reshape(-1, data_val.shape[-1])).to(device)


    dataset_pdc_train = TensorDataset(torch.from_numpy(A_pdc_train).to(device), feature_real_train, feature_imag_train,
                                torch.from_numpy(label_train).to(device))

    dataset_pdc_valid = TensorDataset(torch.from_numpy(A_pdc_valid).to(device), feature_real_valid, feature_imag_valid,
                                torch.from_numpy(label_val).to(device))

    train_loader = DataLoader(dataset_pdc_train, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(dataset_pdc_valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def train_test_split(train_sub, val_sub, feature_de, Adj, label_repeat):

    data_train = feature_de[list(train_sub), :, :]

    data_val = feature_de[list(val_sub), :, :]

    label_train = np.tile(label_repeat, len(train_sub))
    label_val = np.tile(label_repeat, len(val_sub))

    A_pdc_train, A_pdc_valid = Adj[train_sub].reshape([-1, 30, 30]),\
                               Adj[val_sub].reshape([-1, 30, 30])

    return data_train, A_pdc_train, label_train, data_val, A_pdc_valid, label_val


def train_valid(model, optimizer, epochs, train_loader, valid_loader, writer=None, **kwargs):

    criterion = nn.CrossEntropyLoss()

    args = kwargs['args']

    epochs_f1, epochs_loss, epochs_metrics, conf_mat_epochs = [], [], [], []

    met_calc = Metrics(num_class=9)

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)

    if args.label_encoding:
        Loss = loss_fucntion_2(distance_metric=args.distance_metric, dist_features=args.proto_dim)
    # Loss = loss_fucntion_2(distance_metric='L2', dist_features=128)

    best_f1, best_err, early_stopping, best_loss = 0, np.inf, 0, 0

    epoch_grads = {}

    for epoch in tqdm(range(epochs)):

        model.train()

        loss_train, train_correct = 0.0, 0.0

        for i, (graph, X_real, X_imag, label) in enumerate(train_loader):
            start_time = time.time()

            ####################
            # Train
            ####################
            count  = 0.0
            X_real, X_imag = X_real.reshape([-1, 30, 5]), X_imag.reshape([-1, 30, 5])
            preds = model(X_real, X_imag, graph)
            # new loss definition
            # train_loss, pred_label= loss_function(criterion, preds, labels, label, beta=beta, test_flag=True)
            if args.label_encoding:
                train_loss, pred_label= loss_function(criterion, preds, label, distance_metric=args.distance_metric)
            else:
                train_loss, pred_label = Loss(preds, label)

            loss_train += train_loss.detach().item()
            train_correct += (pred_label.squeeze() == label).sum().detach().item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()

        loss_valid = 0.0

        pred_, label_ = [], []

        with torch.no_grad():

            for i, (graph, X_real, X_imag, label) in enumerate(valid_loader):
                start_time = time.time()
                labels = label_encoding(label)
                ####################
                # Valid
                ####################

                X_real, X_imag = X_real.reshape([-1, 30, 5]), X_imag.reshape([-1, 30, 5])
                preds = model(X_real, X_imag, graph)
                if args.label_encoding:
                    valid_loss, pred_label = loss_function(criterion, preds, label,
                                                           distance_metric=args.distance_metric)
                else:
                    valid_loss, pred_label = Loss(preds, label)

                loss_valid += valid_loss.detach().item()
                pred_ += pred_label
                label_ += label.tolist()

        final_metrics = met_calc.compute_metrics(torch.tensor([pred_]).to(device),
                                 torch.tensor([label_]).to(device))


        if writer:

            epochs_metrics.append(final_metrics)

            outstrtrain = 'epoch:%d, Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                          (epoch, loss_valid / len(valid_loader), final_metrics[0], final_metrics[1], final_metrics[2],
                           final_metrics[3])

            print(outstrtrain)

            pred_np, label_np = np.array(torch.tensor(pred_).tolist()), np.array(torch.tensor(label_).tolist())

            conf_mat_epochs.append(confusion_matrix(label_np, pred_np))

            writer.add_scalars('Loss', {'Train': loss_train / len(train_loader),
                                        'Validation': loss_valid / len(valid_loader)}, epoch)

            writer.add_scalars("Accuracy", {'Train': train_correct / len(train_loader.dataset),
                                            'Valid': final_metrics[0]}, epoch)

            writer.add_scalar("recall/val", final_metrics[1], epoch)
            writer.add_scalar("precision/val", final_metrics[2], epoch)
            writer.add_scalar("F1 Score", final_metrics[3], epoch)

    if writer:

        return epochs_metrics, conf_mat_epochs, epoch_grads

    else:

        return best_f1

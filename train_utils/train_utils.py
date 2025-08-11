from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
from utils.utils_loss import *
from sklearn.metrics import confusion_matrix
from scipy.signal import hilbert
import numpy as np
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def dataloader(data_train, label_train, A_pdc_train, data_val, label_val,  A_pdc_valid, q, K, batch_size):

    data_train_reshaped = data_train.reshape(-1, data_train.shape[-1]).astype(np.float32)
    data_val_reshaped = data_val.reshape(-1, data_val.shape[-1]).astype(np.float32)

    feature_real_train = torch.FloatTensor(data_train_reshaped).to(device)
    feature_imag_train = torch.FloatTensor(np.imag(hilbert(data_train_reshaped, axis=-1))).to(device)

    feature_real_valid = torch.FloatTensor(data_val_reshaped).to(device)
    feature_imag_valid = torch.FloatTensor(np.imag(hilbert(data_val_reshaped, axis=-1))).to(device)

    dataset_pdc_train = TensorDataset(torch.from_numpy(A_pdc_train.reshape(-1, 5,
                                                       A_pdc_train.shape[4], A_pdc_train.shape[5])).to(device),
                                                       feature_real_train, feature_imag_train,
                                                       torch.from_numpy(label_train).to(device))

    dataset_pdc_valid = TensorDataset(torch.from_numpy(A_pdc_valid.reshape(-1, 5,
                                                       A_pdc_valid.shape[4], A_pdc_valid.shape[5])).to(device),
                                                       feature_real_valid, feature_imag_valid,
                                                       torch.from_numpy(label_val).to(device))

    train_loader = DataLoader(dataset_pdc_train, batch_size=batch_size, shuffle=True)

    valid_loader = DataLoader(dataset_pdc_valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def train_test_split(train_sub, val_sub, feature_de, Adj, label_repeat):

    data_train = feature_de[list(train_sub), :, :]

    data_val = feature_de[list(val_sub), :, :]

    label_train = np.tile(label_repeat, len(train_sub))
    label_val = np.tile(label_repeat, len(val_sub))

    # A_pdc_train, A_pdc_valid = Adj[train_sub].reshape([-1, 30, 30]),\
    #                            Adj[val_sub].reshape([-1, 30, 30])

    A_pdc_train, A_pdc_valid = Adj[train_sub], \
                               Adj[val_sub]
    A_pdc_train, A_pdc_valid = np.transpose(A_pdc_train, axes=[0, 2, 3, 1, 4, 5]),\
                               np.transpose(A_pdc_valid, axes=[0, 2, 3, 1, 4, 5])

    return data_train, A_pdc_train, label_train, data_val, A_pdc_valid, label_val


# Changed signature to accept Loss object
def train_valid(model, optimizer, Loss, epochs, train_loader, valid_loader, writer=None, **kwargs):

    # criterion = nn.CrossEntropyLoss() # Loss object now handles its own criterion if needed or passed at init

    args = kwargs['args']

    epochs_f1, epochs_loss, epochs_metrics, conf_mat_epochs = [], [], [], []

    met_calc = Metrics(num_class=args.num_classes) # Assuming num_classes is consistently 9

    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)


    # Early stopping initialization
    best_val_metric = 0.0 if args.early_stopping_monitor != 'loss' else float('inf')
    epochs_no_improve = 0
    # best_model_state = None # Optional: for restoring best model

    best_f1, best_err, early_stopping, best_loss = 0, np.inf, 0, 0 # These seem like old/alternative early stopping vars, consider removing if redundant

    lambda_max, warmup_epochs = args.gmm_lambda, 10

    for epoch in tqdm(range(epochs)):

        model.train()

        loss_train, train_correct = 0.0, 0.0

        gmm_lambda = lambda_max * min(epoch / warmup_epochs, 1.0)

        for i, (graph, X_real, X_imag, label) in enumerate(train_loader):
            start_time = time.time()

            ####################
            # Train
            ####################
            count  = 0.0
            X_real, X_imag = X_real.reshape([-1, 30, 5]), X_imag.reshape([-1, 30, 5])
            preds = model(X_real, X_imag, graph)

            # Use UnifiedLoss instance
            if args.multi_head_attention:
                train_loss, pred_label, attention = Loss(preds, label)
            else:
                train_loss, pred_label = Loss(preds, label, gmm_lambda)

            if epoch == 0 and i == 0:
                perv_total_loss = train_loss

            if train_loss.item() > perv_total_loss * 1.02:
                gmm_lambda = max(gmm_lambda - gmm_lambda * 0.1, 0.0)

            perv_total_loss = 0.9 * perv_total_loss + 0.1 * train_loss.item()

            loss_train += train_loss.detach().item()
            train_correct += (pred_label.squeeze() == label).sum().detach().item()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # print(model)
        # current_b = model.cheb_conv1.cprelu.item()
        # print(f"Epoch {epoch + 1:3d}/{epochs}: modReLU bias b = {current_b:.6f}")

        scheduler.step()

        model.eval()

        loss_valid = 0.0

        pred_, label_ = [], []

        with torch.no_grad():

            for i, (graph, X_real, X_imag, label) in enumerate(valid_loader):
                start_time = time.time()
                ####################
                # Valid
                ####################

                X_real, X_imag = X_real.reshape([-1, 30, 5]), X_imag.reshape([-1, 30, 5])
                preds = model(X_real, X_imag, graph)

                # Use UnifiedLoss instance
                if args.multi_head_attention:
                    valid_loss, pred_label, attention = Loss(preds, label)
                else:
                    valid_loss, pred_label = Loss(preds, label, gmm_lambda)

                loss_valid += valid_loss.detach().item()
                pred_.append(pred_label) # Collect tensors
                label_.extend(label.tolist()) # Extend list with batch labels

        # pred_ is a list of tensors (predicted labels per batch)
        # label_ is a list of integers (true labels)
        if pred_: # Ensure pred_ is not empty
            all_preds_tensor = torch.cat(pred_).to(device)
        else: # Handle case where validation loader might be empty or no predictions made
            all_preds_tensor = torch.empty(0, dtype=torch.long).to(device)

        all_labels_tensor = torch.tensor(label_, dtype=torch.long).to(device) # label_ is already a flat list of ints

        final_metrics = met_calc.compute_metrics(all_preds_tensor, all_labels_tensor)


        if writer:

            epochs_metrics.append(final_metrics)

            outstrtrain = 'epoch:%d, Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                          (epoch, (loss_valid / len(valid_loader)) if len(valid_loader) > 0 else 0.0, final_metrics[0], final_metrics[1], final_metrics[2],
                           final_metrics[3])

            print(outstrtrain)

            # For confusion matrix, ensure tensors are on CPU and converted to numpy
            conf_mat_preds = all_preds_tensor.cpu().numpy()
            conf_mat_labels = all_labels_tensor.cpu().numpy()
            conf_mat_epochs.append(confusion_matrix(conf_mat_labels, conf_mat_preds))

            writer.add_scalars('Loss', {'Train': (loss_train / len(train_loader)) if len(train_loader) > 0 else 0.0,
                                        'Validation': loss_valid / len(valid_loader)}, epoch)

            writer.add_scalars("Accuracy", {'Train': train_correct / len(train_loader.dataset),
                                            'Valid': final_metrics[0]}, epoch)

            writer.add_scalar("recall/val", final_metrics[1], epoch)
            writer.add_scalar("precision/val", final_metrics[2], epoch)
            writer.add_scalar("F1 Score", final_metrics[3], epoch)

        # Early stopping check
        # final_metrics = [accuracy, recall, precision, f1_score]
        # loss_valid is sum of batch losses for validation

        current_val_metric_for_early_stopping = 0.0
        actual_val_loss_for_early_stopping = loss_valid / len(valid_loader) if len(valid_loader) > 0 else float('inf')

        if args.early_stopping_monitor == 'f1_score':
            current_val_metric_for_early_stopping = final_metrics[3] # F1-score
        elif args.early_stopping_monitor == 'accuracy':
            current_val_metric_for_early_stopping = final_metrics[0] # Accuracy
        elif args.early_stopping_monitor == 'loss':
            current_val_metric_for_early_stopping = actual_val_loss_for_early_stopping

        if args.early_stopping_patience > 0: # Only if early stopping is enabled
            if args.early_stopping_monitor != 'loss': # Higher is better (Accuracy, F1-score)
                if current_val_metric_for_early_stopping - best_val_metric > args.early_stopping_min_delta:
                    best_val_metric = current_val_metric_for_early_stopping
                    epochs_no_improve = 0
                    # if best_model_state is not None: best_model_state = model.state_dict() # Save best model
                    print(f"EarlyStopping: New best {args.early_stopping_monitor}: {best_val_metric:.4f}")
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_fold_{kwargs["fold"]}.pth'))
                else:
                    epochs_no_improve += 1
                    print(f"EarlyStopping: No improvement in {args.early_stopping_monitor} for {epochs_no_improve} epoch(s). Best: {best_val_metric:.4f}, Current: {current_val_metric_for_early_stopping:.4f}")
            else: # Lower is better (loss)
                if best_val_metric - current_val_metric_for_early_stopping > args.early_stopping_min_delta:
                    best_val_metric = current_val_metric_for_early_stopping
                    epochs_no_improve = 0
                    # if best_model_state is not None: best_model_state = model.state_dict() # Save best model
                    print(f"EarlyStopping: New best validation {args.early_stopping_monitor}: {best_val_metric:.4f}")
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_fold_{kwargs["fold"]}.pth'))
                else:
                    epochs_no_improve += 1
                    print(f"EarlyStopping: No improvement in validation {args.early_stopping_monitor} for {epochs_no_improve} epoch(s). Best: {best_val_metric:.4f}, Current: {current_val_metric_for_early_stopping:.4f}")

            if epochs_no_improve >= args.early_stopping_patience:
                print(f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement.")
                # if best_model_state is not None: model.load_state_dict(best_model_state) # Restore best model
                break # Exit the epoch loop


    if writer:

        return epochs_metrics, conf_mat_epochs

    else:

        return best_f1

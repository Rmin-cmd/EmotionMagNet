from optuna import Trial
import os
import numpy as np
import torch
import time
import scipy.io as sio
from utils import load_data
from hermitian import cheb_poly, decomp
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from utils.utils_loss import *
from tqdm import tqdm
from Model.model import ChebNet


class model_parameter_tuning:

    def __init__(self, loss_func, cat_num, root_dir_feat, root_dir_pdc, num_subs, num_folds, epochs):

        # self.q_value = q_value
        # self.learning_rate = learning_rate
        # self.l2 = l2
        # self.model = model
        self.root_dir_feat = root_dir_feat
        self.root_dir_pdc = root_dir_pdc
        self.num_subs = num_subs
        self.num_folds = num_folds
        self.categories = cat_num
        self.A_pdc = sio.loadmat(root_dir_pdc)['data']

        self.A_pdc = np.mean(self.A_pdc, axis=1)

        self.criterion = loss_func

        self.epochs = epochs

        self.met_calc = Metrics(num_class=cat_num)

    def get_dl_mat(self):
        # build dl_mat
        e = 0.05
        g = 0.03
        eps = 1e-7
        dl_mat = [
            [1 - 3 * e - 2 * g, g, g, e, e, e / 3, e / 3, e / 3, eps],
            [g, 1 - 3 * e - 2 * g, g, e, e, e / 3, e / 3, e / 3, eps],
            [g, g, 1 - 3 * e - 2 * g, e, e, e / 3, e / 3, e / 3, eps],
            [e / 3, e / 3, e / 3, 1 - 3 * e, e, eps, eps, eps, e],
            [e / 3, e / 3, e / 3, e, 1 - 4 * e, e / 3, e / 3, e / 3, e],
            [e / 3, e / 3, e / 3, eps, e, 1 - 3 * e - 2 * g, g, g, e],
            [e / 3, e / 3, e / 3, eps, e, g, 1 - 3 * e - 2 * g, g, e],
            [e / 3, e / 3, e / 3, eps, e, g, g, 1 - 3 * e - 2 * g, e],
            [eps, eps, eps, e, e, e / 3, e / 3, e / 3, 1 - 3 * e]
        ]
        return dl_mat

    def objective(self, trial:Trial):

        acc_fold, re_fold, pre_fold, f1_fold = [], [], [], []

        n_per = self.num_subs // self.num_folds

        label_type = 'cls2' if self.categories == 2 else 'cls9'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fuzzy_labels = self.get_dl_mat()

        q_value = trial.suggest_loguniform('q_value', low=1e-5, high=1)

        learning_rate = trial.suggest_loguniform('learning_rate', low=1e-4, high=1e-2)

        l2 = trial.suggest_categorical('l2', choices=[0.01, 1e-5])

        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        K = trial.suggest_categorical('K', choices=[2, 3, 4, 5])

        layers = trial.suggest_categorical('num_layers', choices=[2, 3, 4, 5])

        for fold in tqdm(range(self.num_folds)):

            model = ChebNet(5, K=K, layer=layers, dropout=0).to(device)

            # feature_pdc = sio.loadmat(feature_path)['de_fold'+str(fold)]
            data_dir = os.path.join(self.root_dir_feat, 'de_lds_fold%d.mat' % (fold))
            feature_pdc = sio.loadmat(data_dir)['de_lds']

            label_repeat = load_data.load_srt_de(feature_pdc, True, label_type, 11)

            if fold < self.num_folds - 1:
                val_sub = np.arange(n_per * fold, n_per * (fold + 1))
            else:
                val_sub = np.arange(n_per * fold, n_per * (fold + 1) - 1)

            train_sub = list(set(np.arange(self.num_subs)) - set(val_sub))

            data_train = feature_pdc[list(train_sub), :, :].reshape(-1, feature_pdc.shape[-1])
            data_val = feature_pdc[list(val_sub), :, :].reshape(-1, feature_pdc.shape[-1])

            label_train = np.tile(label_repeat, len(train_sub))
            label_val = np.tile(label_repeat, len(val_sub))

            # train_featrue, valid_feature, test_feature = feature_pdc[train_mask[:, fold], :, :],\
            #                                              feature_pdc[valid_mask[:, fold], :, :],\
            #                                              feature_pdc[test_mask[:, fold], :, :]
            #
            # train_label, valid_label, test_label = label_pdc[train_mask[:, fold]],\
            #                                        label_pdc[valid_mask[:, fold]],\
            #                                        label_pdc[test_mask[:, fold]]

            A_pdc_train, A_pdc_valid = self.A_pdc[train_sub].reshape([-1, 30, 30]), \
                                       self.A_pdc[val_sub].reshape([-1, 30, 30])

            # data_resample_train, y_resample_train = under_sampling(list(zip(data_train, A_pdc_train)), label_train, ratio=0.01)
            # data_train, A_pdc_train = list(zip(*data_resample_train))
            # data_resample_valid, y_resample_valid = under_sampling(list(zip(data_val, A_pdc_valid)), label_val, ratio=0.01)
            # data_val, A_pdc_valid = list(zip(*data_resample_valid))

            feature_real_train, feature_imag_train = torch.FloatTensor(np.array(data_train)).to(device), \
                                                     torch.FloatTensor(np.array(data_train)).to(device)

            feature_real_valid, feature_imag_valid = torch.FloatTensor(np.array(data_val)).to(device), \
                                                     torch.FloatTensor(np.array(data_val)).to(device)

            her_mat_train = np.array([decomp(data, q_value, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
                                      for data in A_pdc_train])

            her_mat_valid = np.array([decomp(data, q_value, norm=True, laplacian=True, max_eigen=2, gcn_appr=True)
                                      for data in A_pdc_valid])

            cheb_graph_train = np.array([cheb_poly(her_mat_train[i], K) for i in range(her_mat_train.shape[0])])

            cheb_graph_valid = np.array([cheb_poly(her_mat_valid[i], K) for i in range(her_mat_valid.shape[0])])

            dataset_pdc_train = TensorDataset(torch.from_numpy(cheb_graph_train).to(device), feature_real_train,
                                              feature_imag_train,
                                              torch.from_numpy(label_train).to(device))

            dataset_pdc_valid = TensorDataset(torch.from_numpy(cheb_graph_valid).to(device), feature_real_valid,
                                              feature_imag_valid,
                                              torch.from_numpy(label_val).to(device))

            train_loader = DataLoader(dataset_pdc_train, batch_size=batch_size, shuffle=True)

            valid_loader = DataLoader(dataset_pdc_valid, batch_size=batch_size, shuffle=False)

            opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)

            train_correct, train_rec, train_pre, train_f1, loss_train = 0.0, 0.0, 0.0, 0.0, 0.0

            valid_acc, valid_rec, valid_pre, valid_f1, loss_valid = 0.0, 0.0, 0.0, 0.0, 0.0

            met = []

            for epoch in tqdm(range(self.epochs)):

                model.train()

                loss_train, train_correct = 0.0, 0.0

                for i, (graph, X_real, X_imag, label) in enumerate(train_loader):
                    start_time = time.time()
                    ####################
                    # Train
                    ####################
                    count = 0.0

                    labels_prob = torch.tensor([self.fuzzy_labels[y] for y in label])

                    X_real, X_imag = X_real.reshape([-1, 30, 5]), X_imag.reshape([-1, 30, 5])
                    preds = model(X_real, X_imag, graph)
                    train_loss = self.criterion(preds, label.to(torch.int64))  # labels.to(torch.int64)
                    loss_train += train_loss.detach().item()
                    pred_label = preds.max(dim=1)[1]
                    train_correct += (pred_label == label).sum().detach().item()
                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()

                model.eval()

                loss_valid = 0.0

                pred_, label_ = [], []

                with torch.no_grad():

                    for i, (graph, X_real, X_imag, label) in enumerate(valid_loader):
                        start_time = time.time()
                        ####################
                        # Valid
                        ####################
                        count = 0.0

                        labels_prob = torch.tensor([self.fuzzy_labels[y] for y in label])

                        X_real, X_imag = X_real.reshape([-1, 30, 5]), X_imag.reshape([-1, 30, 5])
                        preds = model(X_real, X_imag, graph)
                        valid_loss = self.criterion(preds, label.to(torch.int64))
                        loss_valid += valid_loss.detach().item()
                        pred_label = preds.max(dim=1)[1].tolist()
                        pred_ += pred_label
                        label_ += label.tolist()
                        # train_correct += (pred_label == label).sum().detach().item()

                final_metrics = self.met_calc.compute_metrics(torch.tensor([pred_]).to(device),
                                                         torch.tensor([label_]).to(device))
                met.append(final_metrics)

                outstrtrain = 'epoch:%d, Valid loss: %.6f, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % (
                epoch,
                loss_valid / len(valid_loader), final_metrics[0], final_metrics[1], final_metrics[2], final_metrics[3])

                # print(outstrtrain)

            acc_fold.append(np.mean(met, axis=0)[0]), re_fold.append(np.mean(met, axis=0)[1]), pre_fold.append(
                np.mean(met, axis=0)[2]),
            f1_fold.append(np.mean(met, axis=0)[3])

            trial.report(np.mean(met, axis=0)[3], fold)

        # print(
        #     'folds accuracy: %.3f ± %.3f, folds recall: %.3f ± %.3f, folds precision: %.3f ± %.3f, folds F1-Score: %.3f ± %.3f' %
        #     (
        #     np.mean(acc_fold), np.std(acc_fold), np.mean(re_fold), np.std(re_fold), np.mean(pre_fold), np.std(pre_fold),
        #     np.mean(f1_fold), np.std(f1_fold)))

        return np.mean(f1_fold)


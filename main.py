import scipy.io as sio
import torch.optim as optim
from utils import load_data
import os
from torch.utils.tensorboard import SummaryWriter
from Model_magnet.Magnet_model_2 import ChebNet
from train_utils.train_utils import *
from scipy.stats import beta
from tqdm import tqdm
import numpy as np
import torch


def main(args):

    n_per = args.n_subs // args.n_folds

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class_names = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']

    # data_path = os.path.join(os.getcwd(), 'preprocessed_data', 'preprocessed_connectivity', 'processed_conn_30_mod_4.mat')
    data_path = r'C:\Users\alajv\PycharmProjects\FullyComplexValuedMagnet\preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_4.mat'

    A_pdc = sio.loadmat(data_path)['data']

    acc_fold, recall_fold, precision_fold, f1_score_fold = [], [], [], []

    for fold in tqdm(range(args.n_folds)):

        # root_dir = os.path.join(os.getcwd(), 'preprocessed_data', 'preprocessed_feature', 'smooth_preprocessed_28')
        root_dir = r'C:\Users\alajv\PycharmProjects\FullyComplexValuedMagnet\preprocessed_data\preprocessed_feature\smooth_preprocessed_28'
        data_dir = os.path.join(root_dir, 'de_lds_fold%d.mat' % (fold))
        feature_pdc = sio.loadmat(data_dir)['de_lds']

        label_repeat = load_data.load_srt_de(feature_pdc, True, args.label_type, 11)

        if fold < args.n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1) - 1)

        train_sub = list(set(np.arange(args.n_subs)) - set(val_sub))

        data_train, A_pdc_train, label_train, data_test, A_pdc_test, label_test = train_test_split(
            train_sub,
            val_sub, feature_pdc,
            A_pdc, label_repeat)

        train_loader, valid_loader = dataloader(data_train, label_train, A_pdc_train, data_test, label_test, A_pdc_test, q=args.q,
                                                K=args.K, batch_size=args.batch_size)
        if args.multi_head_attention:
            # TODO: Add multiple head attention mechanism model here
            continue
        else:
            model = ChebNet(5, args=args).to(device)

        print("number of trainable parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        writer = SummaryWriter(log_dir=f"runs/FCMagnet_zero_2/fold_{fold}")

        met_epochs, conf_mat_epochs, epoch_grads = train_valid(model, optimizer, epochs=args.epochs,
                                                               train_loader=train_loader,
                                                               valid_loader=valid_loader, writer=writer,
                                                               args=args)

        fig = show_confusion(np.mean(conf_mat_epochs, axis=0), class_names, show=False)

        writer.add_figure("Confusion Matrix", fig)

        acc_fold.append(np.max(met_epochs, axis=0)[0]), recall_fold.append(np.max(met_epochs, axis=0)[1]),
        precision_fold.append(np.max(met_epochs, axis=0)[2]), f1_score_fold.append(np.max(met_epochs, axis=0)[3])

        outstrtrain = 'fold:%d, accuracy: %.3f, recall:%.3f, precision:%.3f, F1-score:%.3f' % \
                      (fold, acc_fold[fold], recall_fold[fold], precision_fold[fold], f1_score_fold[fold])

        print(outstrtrain)


    print('folds accuracy: %.3f ± %.3f, folds recall: %.3f ± %.3f, folds precision: %.3f ± %.3f, folds F1-Score: %.3f ± %.3f' %
          (np.mean(acc_fold), np.std(acc_fold), np.mean(recall_fold), np.std(recall_fold), np.mean(precision_fold),
           np.std(precision_fold), np.mean(f1_score_fold), np.std(f1_score_fold)))
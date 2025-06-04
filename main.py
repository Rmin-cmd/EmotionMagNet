import scipy.io as sio
import torch.optim as optim
from utils import load_data
import os
from torch.utils.tensorboard import SummaryWriter
from Model_magnet.Magnet_model_2 import ChebNet
from Model_magnet.encoding_loss_function import UnifiedLoss # Import UnifiedLoss
from train_utils.train_utils import *
from scipy.stats import beta
from tqdm import tqdm
import numpy as np
import torch


def main(args):

    n_per = args.n_subs // args.n_folds

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class_names = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']

    A_pdc = sio.loadmat(args.data_path)['data']

    acc_fold, recall_fold, precision_fold, f1_score_fold = [], [], [], []

    all_subject_indices = np.arange(args.n_subs)
    # Optional: np.random.shuffle(all_subject_indices) if folds should be random across runs

    for fold in tqdm(range(args.n_folds)):

        data_dir = os.path.join(args.feature_root_dir, 'de_lds_fold%d.mat' % (fold))
        feature_pdc = sio.loadmat(data_dir)['de_lds']

        label_repeat = load_data.load_srt_de(feature_pdc, True, args.label_type, 11)

        start_index = (args.n_subs // args.n_folds) * fold
        end_index = (args.n_subs // args.n_folds) * (fold + 1)
        if fold == args.n_folds - 1:
            val_sub = all_subject_indices[start_index:]
        else:
            val_sub = all_subject_indices[start_index:end_index]
        train_sub = np.setdiff1d(all_subject_indices, val_sub)

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
            model = ChebNet(args.in_channels, args=args).to(device)

        # Instantiate UnifiedLoss here
        criterion_for_loss = torch.nn.CrossEntropyLoss() # Define criterion to be passed
        loss_type_arg = 'label_encoding' if args.label_encoding else 'prototype'
        num_classes = 9 # Or derive from data/args if it can change
        label_encoding_temperature = 1.0

        Loss_fn = UnifiedLoss(
            loss_type=loss_type_arg,
            num_classes=num_classes,
            distance_metric=args.distance_metric,
            dist_features=args.proto_dim,
            temperature=label_encoding_temperature,
            gmm_lambda=args.gmm_lambda,
            criterion=criterion_for_loss
        ).to(device)

        # Print number of trainable parameters for model and loss
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        loss_params = sum(p.numel() for p in Loss_fn.parameters() if p.requires_grad)
        print(f"Number of trainable model parameters: {model_params}")
        print(f"Number of trainable loss parameters: {loss_params}")
        print(f"Total trainable parameters: {model_params + loss_params}")


        optimizer = optim.Adam(
            list(model.parameters()) + list(Loss_fn.parameters()),
            lr=args.learning_rate,
            weight_decay=args.l2_normalization # Add this line
        )

        writer = SummaryWriter(log_dir=f"runs/{loss_type_arg}_gmmLambda{args.gmm_lambda}_proto_dim_{args.proto_dim}"
                                       f"/fold_{fold}")

        # Pass Loss_fn to train_valid (Remove epoch_grads from returned values if it's truly gone)
        # met_epochs, conf_mat_epochs, epoch_grads = train_valid(model, optimizer, epochs=args.epochs,
        met_epochs, conf_mat_epochs = train_valid(model, optimizer, Loss_fn, epochs=args.epochs,
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
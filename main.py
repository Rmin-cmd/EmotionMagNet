import scipy.io as sio
import torch.optim as optim
from utils import load_data
import os
import torch.nn as nn
# from Model_magnet.encoding_loss_function2 import UnifiedLoss
from Model_magnet.encoding_loss_function2 import UnifiedLoss
from torch.utils.tensorboard import SummaryWriter
from Model_magnet.Magnet_model_2 import ChebNet as ChebNet_Original
from Model_magnet.REAL_Cheb import ChebNetReal
from Model_magnet.gcn_models import SAGENet, GCNNet, GINNet, APPNPNet, GATNet
from Model_magnet.Magnet_model_multi_head_attention import ChebNet as ChebNet_MultiHead
from train_utils.train_utils import *
from tqdm import tqdm
import numpy as np
import torch
from GCN_pyg import preprocess_pdc
import random

from datetime import datetime
today_date = str(datetime.now())
today_date = today_date[:today_date.find('.')]
today_date = today_date.replace(' ', '/')
today_date = today_date.replace(':', '_')
exp = 'experiment_' + today_date

def main(args):
    args.save_dir = 'saved_models'
    os.makedirs(args.save_dir, exist_ok=True)
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    n_per = args.n_subs // args.n_folds

    num_windows = 11

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    class_names = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']

    acc_fold, recall_fold, precision_fold, f1_score_fold = [], [], [], []

    all_subject_indices = np.arange(args.n_subs)
    # Optional: np.random.shuffle(all_subject_indices) if folds should be random across runs

    for fold in tqdm(range(args.n_folds)):

        data_dir = os.path.join(args.feature_root_dir, 'de_lds_fold%d.mat' % (fold))
        feature_pdc = sio.loadmat(data_dir)['de_lds']

        A_pdc = sio.loadmat(args.data_path)['data']

        # A_pdc = preprocess_pdc(A_pdc, trials_per_subject=28 * 11).reshape(A_pdc.shape)

        if args.num_classes == 9:
            label_type = "cls9" # Or derive from data/args if it can change
        elif args.num_classes == 2:
            feature_pdc = feature_pdc.reshape([feature_pdc.shape[0], -1, num_windows, feature_pdc.shape[2]])
            vid_sel = list(range(12))
            vid_sel.extend(list(range(16, 28)))
            feature_pdc = feature_pdc[:, vid_sel, :, :]  # sub, vid, n_channs, n_points
            A_pdc = A_pdc[:, :, vid_sel, :, :, :]
            feature_pdc = feature_pdc.reshape([feature_pdc.shape[0], -1, feature_pdc.shape[3]])
            label_type = "cls2"
        else:
            raise ValueError("number of specified classes should be 2 or 9")

        label_repeat = load_data.load_srt_de(feature_pdc, True, label_type, num_windows)

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

        # Model selection
        if args.simple_gcn_model is None:
            if args.multi_head_attention:
                print(f"INFO: Using Multi-Head Attention GCN model for fold {fold}.")
                model = ChebNet_MultiHead(in_c=args.in_channels, args=args)
            else:
                model = ChebNet_Original(in_c=args.in_channels, args=args)
        else:
            if args.simple_gcn_model == "GCN":
                model = GCNNet(in_c=args.in_channels, args=args)
            elif args.simple_gcn_model == "Cheb_real":
                model = ChebNetReal(in_c=args.in_channels, args=args)
            elif args.simple_gcn_model == "GIN":
                model = GINNet(in_c=args.in_channels, args=args)
            elif args.simple_gcn_model == "GAT":
                model = GATNet(in_c=args.in_channels, args=args)
            elif args.simple_gcn_model == "APPNP":
                model = APPNPNet(in_c=args.in_channels, args=args)
            elif args.simple_gcn_model == "SAGE":
                model = SAGENet(in_c=args.in_channels, args=args)

        if args.T4_2_flag:
            model = nn.DataParallel(model, device_ids=[0, 1])

        model = model.to(device)

            # Instantiate UnifiedLoss here
        criterion_for_loss = torch.nn.CrossEntropyLoss() # Define criterion to be passed
        if args.simple_gcn_model is not None or args.simple_magnet:
            loss_type_arg = 'simple'
        else:
            loss_type_arg = 'label_encoding' if args.label_encoding else 'prototype'

        label_encoding_temperature = 0.5

        Loss_fn = UnifiedLoss(
            loss_type=loss_type_arg,
            num_classes=args.num_classes,
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
            weight_decay=args.l2_normalization
        )

        writer = SummaryWriter(log_dir=f"runs/{exp}/num_classes_{args.num_classes}_{loss_type_arg}"
                                       f"_gmmLambda{args.gmm_lambda}_proto_dim_{args.proto_dim}"
                                       f"_multi_head_{args.multi_head_attention}_q_{args.q}"
                                       f"/fold_{fold}")

        # Pass Loss_fn to train_valid
        met_epochs, conf_mat_epochs = train_valid(model, optimizer, Loss_fn, epochs=args.epochs,
                                                               train_loader=train_loader,
                                                               valid_loader=valid_loader, writer=writer,
                                                               args=args, fold=fold)

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
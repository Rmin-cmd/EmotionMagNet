import argparse
from main import main


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Emotion Recognition Using Magnetic Graph Convolution")

    # Data paths
    parser.add_argument("--data_path", default='./data/connectivity.mat',
                        help="Path to the connectivity data MAT file")
    parser.add_argument("--feature_root_dir", default='./data/features',
                        help="Root directory for feature data")

    # hyperparameters
    parser.add_argument("--n_subs", default=123, type=int, help="number of subjects in the dataset")
    parser.add_argument("--n_folds", default=10, type=int, help="specify the number of folds")
    parser.add_argument("--label_type", default="cls9", type=str, help="specify the number of classes in the dataset")
    parser.add_argument("--epochs", default=50, type=int, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, type=int, help="learning rate of the specified model")
    parser.add_argument("--dropout", default=0.2, type=int, help="specify the dropout ratio")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size of the dataloader")
    parser.add_argument("--q", default=0.01, type=int, help="magnetic q-value specification")
    parser.add_argument("--l2_normalization", default=1e-7, type=int,
                        help="apply l2 normalization to the optimization module")
    parser.add_argument("--proto_dim", default=128, type=int, help="dimension of the extracted prototypes")
    parser.add_argument("--K", default=3, type=int, help="Number of chebyshev polynomials")
    parser.add_argument("--num_filter", default=2, type=int, help="number of graph convolution layers")
    parser.add_argument("--distance_metric", default="L2", type=str,
                        help="define the distance metric between L1, L2 and orthogonal")
    parser.add_argument("--num_heads", default=4, type=int, help="number of attention heads for each frequency band")
    parser.add_argument("--in_channels", type=int, default=5, help="Number of input channels/features for the model")
    parser.add_argument("--gmm_lambda", type=float, default=0.01, help="Lambda for GMM prototype regularization in prototype loss")

    # Early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of epochs to wait for improvement before early stopping (0 to disable)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Minimum change in monitored quantity to qualify as an improvement")
    parser.add_argument("--early_stopping_monitor", type=str, default="f1_score", choices=["f1_score", "accuracy", "loss"], help="Metric to monitor for early stopping (f1_score, accuracy, or loss)")

    # Different Model types
    parser.add_argument("--FFT-or-not", action='store_false',
                        help="imaginary part calculated from the fourier transform")
    parser.add_argument("--label_encoding", action='store_true',
                        help="encode labels on the 2-d plane of the valence-arousal")
    parser.add_argument("--simple_attention", action="store_true",
                        help="a simple attention applied on the features")
    parser.add_argument("--multi_head_attention", action="store_true",
                        help="apply an attention mechanism on the specified model")
    parser.add_argument("--GMM", action="store_true",
                        help="whenever prototype selection specified for the mentioned prototypes")

    args = parser.parse_args()

    main(args)




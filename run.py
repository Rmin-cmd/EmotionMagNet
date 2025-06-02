import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser("Emotion Recognition Using Magnetic Graph Convolution")

    # hyperparameters
    parser.add_argument("--n_subs", default=123, help="number of subjects in the dataset")
    parser.add_argument("--n_folds", default=10, help="specify the number of folds")
    parser.add_argument("--label_type", default="cls_9", help="specify the number of classes in the dataset")
    parser.add_argument("--epochs", default=50, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.01, help="learning rate of the specified model")
    parser.add_argument("--dropout", default=0.2, help="specify the dropout ratio")
    parser.add_argument("--batch_size", default=64, help="batch size of the dataloader")
    parser.add_argument("--q_value", default=0.01, help="magnetic q-value specification")
    parser.add_argument("--l2_normalization", default=5e-4,
                        help="apply l2 normalization to the optimization module")
    parser.add_argument("--proto_dim", default=128, help="dimension of the extracted prototypes")
    parser.add_argument("--K", default=3, help="Number of chebyshev polynomials")
    parser.add_argument("--num_filter", default=2, help="number of graph convolution layers")
    parser.add_argument("--distance_metric", default="L2",
                        help="define the distance metric between L1, L2 and orthogonal")


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





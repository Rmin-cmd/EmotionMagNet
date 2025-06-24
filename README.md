# EmotionMagNet: Emotion Recognition using Complex Graph Convolutional Networks

## Overview

EmotionMagNet is a research project focused on emotion recognition from physiological signals (e.g., EEG) using graph convolutional networks (GCNs). This repository explores the use of complex-valued GCNs, including concepts inspired by Magnetic Graph Networks, and incorporates advanced mechanisms like multi-head attention within graph convolutions and attention across different frequency band representations.

The models aim to classify emotions based on features extracted from subjects and their functional connectivity (e.g., Partial Directed Coherence - PDC) across different frequency bands.

## Features

*   **Complex-Valued Graph Convolutions:** Utilizes `complextorch` for building GCN layers that operate on complex numbers, potentially capturing richer phase and magnitude information.
*   **Chebyshev Polynomial Basis:** Implements ChebNet-style graph convolutions.
*   **Magnetic Laplacian:** Option to use a magnetic Laplacian (`q` parameter) that incorporates edge directionality information into the graph structure.
*   **Two Main Model Architectures:**
    1.  `ChebNet_Original`: A complex GCN model.
    2.  `ChebNet_MultiHead`: An advanced GCN model featuring:
        *   **Multi-Head Attention in GCN Layers:** Each ChebConv layer can use multiple attention heads to learn diverse node representations.
        *   **Frequency Band Attention:** Processes multiple graph structures (derived from different frequency bands) independently through the GCN, and then uses a self-attention mechanism to weigh and combine these frequency-specific representations.
*   **Attention Mechanisms:**
    *   Option for simple attention within GCN layers (`--simple_attention`).
    *   Multi-head attention as described above (`--multi_head_attention`).
*   **Loss Function Options:**
    *   `--label_encoding`: Uses fixed complex embeddings in the output space as targets.
    *   Prototype-based loss (default): Learns complex-valued prototypes for each class, with an optional GMM-inspired regularization term (`--gmm_lambda`).
*   **Regularization:** Dropout, L2 Weight Decay.
*   **Early Stopping:** Monitor validation metrics to prevent overfitting and stop training when performance plateaus.
*   **TensorBoard Logging:** For visualizing training progress, loss, metrics, and confusion matrices.

## Setup / Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd EmotionMagNet
    ```
2.  **Create a Python environment:**
    Recommended to use a virtual environment (e.g., conda or venv).
    ```bash
    conda create -n emotionmagnet python=3.8
    conda activate emotionmagnet
    ```
3.  **Install dependencies:**
    (A `requirements.txt` file would be ideal here. For now, listing common ones based on the code):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Adjust for your CUDA version or CPU
    pip install scipy numpy tqdm scikit-learn matplotlib tensorboard torchmetrics
    pip install complextorch
    ```
    *Note: `complextorch` might need specific installation steps depending on its source.*

## Dataset

The models expect data in `.mat` files:

*   **Connectivity Data (`--data_path`):** A `.mat` file containing adjacency matrices (e.g., PDC matrices). Expected to be a dictionary where the key 'data' holds a NumPy array of shape `(num_subjects, num_bands, num_nodes, num_nodes)` or similar, which is then processed into `(num_subjects_in_fold * num_trials_per_subject, num_bands, num_nodes, num_nodes)`.
*   **Feature Data (`--feature_root_dir`):** A directory containing `.mat` files for each fold (e.g., `de_lds_fold0.mat`, `de_lds_fold1.mat`, ...). Each file should contain features (e.g., Differential Entropy - DE) in a dictionary where the key 'de_lds' holds a NumPy array, typically reshaped to `(num_subjects_in_fold, num_trials_per_subject, num_nodes, num_features_per_node_timeseries)`.

The current setup seems tailored for datasets like SEED or DEAP where subject-specific trials, multiple frequency bands for connectivity, and node features are available.

## Usage

To train a model, use `run.py`. You can see all available options using:
```bash
python run.py --help
```

**Key Command-Line Arguments:**

*   **Data Paths:**
    *   `--data_path`: Path to the connectivity data `.mat` file. (Default: `./data/connectivity.mat`)
    *   `--feature_root_dir`: Root directory for feature data `.mat` files. (Default: `./data/features`)
*   **Model Selection & Architecture:**
    *   `--multi_head_attention`: Use the `ChebNet_MultiHead` model with frequency attention. If not set, `ChebNet_Original` is used.
    *   `--num_heads`: Number of attention heads in `ChebNet_MultiHead`'s ChebConv layers. (Default: 4)
    *   `--simple_attention`: Enable simple attention within ChebConv layers (applies if not using `--multi_head_attention` or for `cheb_conv1` in `ChebNet_MultiHead`).
    *   `--in_channels`: Number of input node features. (Default: 5)
    *   `--num_filter`: Number of output filters per head in ChebConv layers. (Default: 2)
    *   `--K`: Number of Chebyshev polynomial orders. (Default: 3)
    *   `--q`: Magnetic q-value for Laplacian. (Default: 0.01)
*   **Training Hyperparameters:**
    *   `--epochs`: Number of training epochs. (Default: 50)
    *   `--learning_rate`: Learning rate. (Default: 0.01, consider reducing for complex models e.g., 1e-3, 1e-4)
    *   `--batch_size`: Batch size. (Default: 64)
    *   `--dropout`: Dropout ratio. (Default: 0.2)
    *   `--l2_normalization`: L2 weight decay for the optimizer. (Default: 1e-5)
*   **Loss Function & Prototypes:**
    *   `--label_encoding`: Use fixed label encodings instead of prototype loss.
    *   `--proto_dim`: Dimension of learned prototypes (if not using label encoding). (Default: 128)
    *   `--distance_metric`: Distance metric for prototype loss ('L1', 'L2', 'orth'). (Default: 'L2')
    *   `--gmm_lambda`: Lambda for GMM prototype regularization. (Default: 0.01)
*   **Early Stopping:**
    *   `--early_stopping_patience`: Epochs to wait for improvement. (Default: 10, 0 to disable)
    *   `--early_stopping_min_delta`: Min improvement delta. (Default: 0.001)
    *   `--early_stopping_monitor`: Metric to monitor ('f1_score', 'accuracy', 'loss'). (Default: 'f1_score')
*   **Data & Cross-Validation:**
    *   `--n_subs`: Total number of subjects in the dataset. (Default: 123)
    *   `--n_folds`: Number of cross-validation folds. (Default: 10)
    *   `--label_type`: Type of classification task (e.g., 'cls9' for 9 classes). (Default: "cls9")

**Example Run (Multi-Head Attention Model):**
```bash
python run.py \
    --data_path /path/to/your/processed_conn.mat \
    --feature_root_dir /path/to/your/de_lds_features/ \
    --multi_head_attention \
    --num_heads 4 \
    --num_filter 16 \
    --learning_rate 0.001 \
    --dropout 0.3 \
    --l2_normalization 1e-5 \
    --early_stopping_patience 15 \
    --early_stopping_monitor f1_score \
    --epochs 100
```

## Model Architectures

*   **`ChebNet_Original` (`Model_magnet/Magnet_model_2.py`):**
    A graph convolutional network using complex ChebConv layers. It averages graph bands before GCN processing. The final layers involve feature averaging then a 1D complex convolution.
*   **`ChebNet_MultiHead` (`Model_magnet/Magnet_model_multi_head_attention.py`):**
    An advanced architecture.
    1.  Its `ChebConv` layers implement multi-head attention internally (multiple attention calculations per layer, outputs concatenated).
    2.  It processes each input graph frequency band independently through these multi-head GCN layers.
    3.  The resulting graph-level embedding for each band (after GCN and global graph pooling) is then passed to a `FrequencySelfAttention` module.
    4.  This attention module learns to weigh the importance of each frequency band.
    5.  The weighted combination of band embeddings is fed to a final complex linear layer for classification.

## Logging

Training progress, losses, and metrics are logged to TensorBoard. Events are typically saved in a `runs/` directory.

## Potential Issues & Debugging

*   **NaN Loss:** If you encounter NaN loss:
    *   Try reducing the learning rate (`--learning_rate`).
    *   Try reducing or setting L2 normalization to zero (`--l2_normalization 0`).
    *   Consider adding gradient clipping (not yet implemented).
    *   Ensure the `ComplexBatchNorm1d` layer in `ChebNet` is active if it helps stabilize.
*   **Non-convergence:**
    *   Experiment with learning rates.
    *   Simplify the model (e.g., `--num_heads 1`, disable frequency attention temporarily by modifying code) to isolate problematic components.
    *   Check data quality and preprocessing.

## References (Placeholder)

*   If based on "Magnetic Graph Network": *Zhang, M., Zhang, S., Li, C., & Li, H. (2021). Magnetic Graph Network. arXiv preprint arXiv:2102.11784.*
*   Original ChebNet: *Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. Advances in neural information processing systems, 29.*
*   Attention Is All You Need (for multi-head attention concepts): *Vaswani, A., et al. (2017). Attention is all you need. Advances in neural information processing systems, 30.*

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your features or fixes, and submit a pull request.

## License

(Placeholder - e.g., MIT License)
This project is licensed under the MIT License - see the LICENSE.md file for details (if one exists).

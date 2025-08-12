import os
import shutil
import numpy as np
from utils.plots import save_aggregated_attention_plots

def test_save_aggregated_attention_plots():
    # 1. Setup
    test_dir = 'test_aggregated_plots'
    num_classes = 4
    num_bands = 5

    # Create dummy data
    attention_arrays = [
        np.random.rand(2, num_classes, num_bands), # batch 1: 2 items
        np.random.rand(2, num_classes, num_bands)  # batch 2: 2 items
    ]
    # Total of 4 samples
    labels = np.array([0, 1, 2, 3])

    epoch = 1
    fold = 0

    # 2. Execute
    save_aggregated_attention_plots(attention_arrays, labels, epoch, fold, base_dir=test_dir)

    # 3. Assert
    expected_dir = os.path.join(test_dir, f'fold_{fold}', f'epoch_{epoch}')
    assert os.path.isdir(expected_dir)

    # We expect one plot per class
    expected_files = [f'aggregated_attention_class_{c}.png' for c in range(num_classes)]

    for f in expected_files:
        assert os.path.isfile(os.path.join(expected_dir, f))

    # 4. Teardown
    shutil.rmtree(test_dir)

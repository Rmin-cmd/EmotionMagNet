import os
import shutil
import numpy as np
from utils.plots import save_attention_heatmaps

def test_save_attention_heatmaps():
    # 1. Setup
    test_dir = 'test_heatmaps'
    # os.makedirs(test_dir, exist_ok=True) # The function should create the dir

    # Create dummy data
    attention_arrays = [
        np.random.rand(2, 4, 5), # batch 1: 2 items, 4 classes, 5 bands
        np.random.rand(2, 4, 5)  # batch 2: 2 items, 4 classes, 5 bands
    ]
    epoch = 1
    fold = 0

    # 2. Execute
    save_attention_heatmaps(attention_arrays, epoch, fold, base_dir=test_dir)

    # 3. Assert
    expected_dir = os.path.join(test_dir, f'fold_{fold}', f'epoch_{epoch}')
    assert os.path.isdir(expected_dir)

    expected_files = [
        'batch_0_item_0.png',
        'batch_0_item_1.png',
        'batch_1_item_0.png',
        'batch_1_item_1.png'
    ]

    for f in expected_files:
        assert os.path.isfile(os.path.join(expected_dir, f))

    # 4. Teardown
    shutil.rmtree(test_dir)

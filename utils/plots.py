import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def save_attention_heatmaps(attention_arrays, epoch, fold, base_dir='attention_heatmaps'):
    """
    Saves heatmaps of attention arrays for a given epoch and fold.

    Args:
        attention_arrays (list): A list of numpy arrays, where each array is a batch of attention weights.
                                 The shape of each array is (batch_size, num_classes, num_bands).
        epoch (int): The current epoch number.
        fold (int): The current fold number.
        base_dir (str): The base directory to save the heatmaps in.
    """
    # Create the directory for the fold and epoch
    save_dir = os.path.join(base_dir, f'fold_{fold}', f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    # Iterate through each batch of attention arrays
    batch_item_count = 0
    for batch_idx, batch_attention in enumerate(attention_arrays):
        # Iterate through each attention map in the batch
        for item_idx in range(batch_attention.shape[0]):
            attention_map = batch_attention[item_idx]  # Shape: (num_classes, num_bands)

            plt.figure(figsize=(10, 8))
            sns.heatmap(attention_map, cmap='viridis')
            plt.title(f'Attention Heatmap - Fold {fold}, Epoch {epoch}, Batch {batch_idx}, Item {item_idx}')
            plt.xlabel('Bands')
            plt.ylabel('Classes')

            # Save the figure
            save_path = os.path.join(save_dir, f'batch_{batch_idx}_item_{item_idx}.png')
            plt.savefig(save_path)
            plt.close()
            batch_item_count +=1

    print(f"Saved {batch_item_count} attention heatmaps to {save_dir}")

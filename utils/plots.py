import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def save_aggregated_attention_plots(attention_arrays, labels, epoch, fold, base_dir='aggregated_attention_plots'):
    """
    Aggregates attention scores for each class and saves plots for a given epoch and fold.

    Args:
        attention_arrays (list): A list of numpy arrays, where each array is a batch of attention weights.
                                 The shape of each array is (batch_size, num_classes, num_bands).
        labels (np.ndarray): A numpy array of true labels for all samples.
        epoch (int): The current epoch number.
        fold (int): The current fold number.
        base_dir (str): The base directory to save the plots in.
    """
    # Create the directory for the fold and epoch
    save_dir = os.path.join(base_dir, f'fold_{fold}', f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    # Concatenate all attention batches into a single array
    if not attention_arrays:
        print("No attention arrays to process.")
        return

    all_attentions = np.concatenate(attention_arrays, axis=0)

    num_samples, num_classes, num_bands = all_attentions.shape

    # Ensure labels match the number of samples
    if len(labels) != num_samples:
        print(f"Warning: Number of labels ({len(labels)}) does not match number of samples ({num_samples}). Skipping plotting.")
        return

    # Aggregate attention vectors for each class
    aggregated_attentions = [[] for _ in range(num_classes)]
    for i in range(num_samples):
        true_label = labels[i]
        # Get the attention scores from the prototype corresponding to the true label
        attention_vector = all_attentions[i, true_label, :]
        aggregated_attentions[true_label].append(attention_vector)

    # Plot and save the aggregated attention for each class
    for c in range(num_classes):
        if aggregated_attentions[c]:
            # Calculate the average attention vector for the class
            mean_attention = np.mean(aggregated_attentions[c], axis=0)

            plt.figure(figsize=(12, 6))

            # Create a bar chart
            band_indices = np.arange(num_bands)
            plt.bar(band_indices, mean_attention)

            plt.title(f'Aggregated Attention for Class {c} - Fold {fold}, Epoch {epoch}')
            plt.xlabel('Frequency Band')
            plt.ylabel('Average Attention Score')
            plt.xticks(band_indices)
            plt.grid(axis='y', linestyle='--')

            # Save the figure
            save_path = os.path.join(save_dir, f'aggregated_attention_class_{c}.png')
            plt.savefig(save_path)
            plt.close()

    print(f"Saved aggregated attention plots to {save_dir}")

import matplotlib.pyplot as plt
import numpy as np


def plot_on_unit_circle(preds, label_en, labels, predicted_labels):
    """
    Plots the real and imaginary parts of normalized predicted values and corresponding labels on a unit circle.

    Args:
        preds: Complex-valued predictions from the model.
        label_en: Complex-valued encoded labels.
        labels: Ground truth labels (as integers).
        predicted_labels: argmax of the predicted outputs (as integers).
    """
    # Extract real and imaginary components
    pred_real = preds.real.cpu().numpy()
    pred_imag = preds.imag.cpu().numpy()
    label_real = label_en.real.cpu().numpy()
    label_imag = label_en.imag.cpu().numpy()

    # Convert ground truth labels and predicted labels to NumPy arrays for compatibility
    true_labels_np = labels.cpu().numpy()
    predicted_labels_np = predicted_labels.cpu().numpy()

    # Define unique colors for each label
    # unique_labels = np.unique(predicted_labels_np)
    unique_labels = np.array([*range(9)])
    colors = plt.cm.tab10(unique_labels / len(unique_labels))  # Use a colormap (tab10) for distinct colors

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set equal scaling, center the axes
    ax.set_xlim(-5.1, 5.1)
    ax.set_ylim(-5.1, 5.1)
    ax.set_aspect('equal')  # Ensure the circle is not distorted

    # Draw the circle (radius = 1)
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    ax.add_artist(circle)

    # Set the axis labels
    ax.set_xlabel('Valence/Real')
    ax.set_ylabel('Arousal/Imaginary')

    # Draw gridlines or ticks
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Scatter plot for predictions, colored by predicted labels
    # for label in unique_labels:
    #     indices = np.where(predicted_labels_np == label)
    #     ax.scatter(
    #         # pred_real[indices, label], pred_imag[indices, label],
    #         pred_real[0, label], pred_imag[0, label],
    #         label=f'Predicted: {label}',
    #         color=colors[label],
    #         alpha=0.7,
    #         edgecolor='black'
    #     )

    # Scatter plot for true labels (optional, as encoded complex values)
    ax.scatter(
        label_real, label_imag,
        color='red', marker='x', label='Ground Truth Labels'
    )

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # Title
    ax.set_title('Predictions and Labels on Unit Circle')

    plt.pause(1.0)

    plt.ioff()
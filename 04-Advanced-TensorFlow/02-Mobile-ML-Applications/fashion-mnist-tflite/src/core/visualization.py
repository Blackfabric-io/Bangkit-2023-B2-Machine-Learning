import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any

def plot_image(i: int, predictions_array: List[Any], true_label: List[Any], 
              img: List[Any], class_names: List[str]) -> None:
    """Plot a single image with its predictions.
    
    Args:
        i: Index of the image to plot
        predictions_array: Array of model predictions
        true_label: Array of true labels
        img: Array of images
        class_names: List of class names
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    color = 'green' if predicted_label == true_label.numpy() else 'red'
        
    plt.xlabel(f"{class_names[predicted_label]} {100*np.max(predictions_array):2.0f}% ({class_names[true_label]})",
              color=color)

def plot_value_array(i: int, predictions_array: List[Any], 
                    true_label: List[Any]) -> None:
    """Plot the prediction values for a single image.
    
    Args:
        i: Index of the predictions to plot
        predictions_array: Array of model predictions
        true_label: Array of true labels
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(list(range(10)))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array[0], color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array[0])
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue') 
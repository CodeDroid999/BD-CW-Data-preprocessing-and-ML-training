import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the dataset using load_dataset()
dataset = tfds.load('dataset_name', split='train')

# Display 9 images from the dataset
fig = plt.figure(figsize=(10, 10))
for i, example in enumerate(dataset.take(9)):
  # Extract the image and label from the example
  image, label = example['image'], example['label']
  
  # Create a subplot for each image
  ax = fig.add_subplot(3, 3, i+1)
  
  # Remove the x and y ticks from the plot
  ax.set_xticks([])
  ax.set_yticks([])
  
  # Display the image on the plot
  ax.imshow(image.numpy())
  
  # Set the title of the plot to the corresponding label
  ax.set_title(f"Label: {label.numpy()}")

# Show the plot
plt.show()

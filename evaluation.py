import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

# Function to load the generator model architecture from a .json file and weights from a .h5 file
def load_generator(model_json_path, model_weights_path):
    # Load the model architecture from the .json file
    try:
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        generator = model_from_json(model_json)
        
        print(f"Model type: {type(generator)}")  # Debugging: Print the type of the loaded model
        
    except Exception as e:
        raise ValueError(f"Error loading model architecture from {model_json_path}: {e}")

    # Check if generator is a valid Keras model before loading weights
    if hasattr(generator, 'load_weights'):
        # Load the model weights from the .h5 file
        try:
            generator.load_weights(model_weights_path)
        except Exception as e:
            raise ValueError(f"Error loading weights from {model_weights_path}: {e}")
    else:
        raise ValueError(f"The loaded model is not a valid Keras model. Received object of type: {type(generator)}")
    
    return generator

# Function to generate images from random noise
def generate_images(generator, num_images, z_dim):
    z = np.random.normal(size=[num_images, z_dim])  # Generate random noise
    generated_images = generator.predict(z)  # Generate images using the loaded model
    return generated_images

# Function to visualize the generated images (5x5 grid)
def visualize_images(generated_images, h=28, w=28, n=5):
    I_generated = np.empty((h * n, w * n))
    
    for i in range(n):
        for j in range(n):
            I_generated[i * h:(i + 1) * h, j * w:(j + 1) * w] = generated_images[i * n + j, :].reshape(h, w)

    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(I_generated, cmap='gray')
    plt.show()

# Main function
if __name__ == '__main__':
    # Paths to the .json and .h5 files
    model_json_path = r"C:\Users\SUNYLoaner\Downloads\final_generator_model.json" # Path to the model architecture file
    model_weights_path = r"C:\Users\SUNYLoaner\Downloads\final_generator_model.h5"  # Path to the model weights file
    
    # Load the generator model
    generator = load_generator(model_json_path, model_weights_path)
    
    # Hyperparameters (ensure these match the values used during training)
    latent_size = 100  # Latent space size (input to the generator)
    
    # Generate images
    num_images = 25  # 5x5 grid requires 25 images
    generated_images = generate_images(generator, num_images, latent_size)
    
    # Visualize the generated images in a 5x5 grid
    visualize_images(generated_images)

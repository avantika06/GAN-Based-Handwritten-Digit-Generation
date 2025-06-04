

# Generative Adversarial Networks (GANs) for handwritten digits 

## Introduction  
This project implements **Generative Adversarial Networks (GANs)** to generate synthetic images of handwritten digits using the **MNIST dataset**. The GAN consists of a **generator** and a **discriminator**, which are trained through adversarial learning to improve the quality of generated images.  

## Features  
- **GAN Implementation**: Trained on the MNIST dataset to generate realistic handwritten digits.  
- **Two Generator Models**:  
  - **Model 1**: Standard GAN architecture with fully connected layers and ReLU activations.  
  - **Model 2**: Enhanced architecture with **batch normalization** and **Leaky ReLU** for better stability.  
- **Discriminator Model**: A fully connected network that classifies real vs. generated images.  
- **Training Process**: Uses **Adam optimizer**, **Binary Cross-Entropy Loss**, and **data normalization** for stable learning.  
- **Visualization**: Outputs generated images and training loss curves to analyze model performance.  

## Dataset  
The **MNIST dataset** consists of **60,000 training images** and **10,000 test images** of handwritten digits (0-9). Each image is **28x28 grayscale**. The dataset is preprocessed by normalizing pixel values for efficient training.  

## Model Architectures  

### **Generator (Model 1)**  
- **Input**: 100-dimensional latent space  
- **Hidden Layers**:  
  - 256 neurons (ReLU)  
  - 512 neurons (ReLU)  
  - 1024 neurons (ReLU)  
- **Output**: 784-dimensional (28x28 image, Tanh activation)  

### **Generator (Model 2)**  
- **Enhancements**: Batch Normalization + Leaky ReLU  
- **Same layer sizes as Model 1**  

### **Discriminator**  
- **Input**: 784-dimensional (flattened MNIST image)  
- **Hidden Layers**:  
  - 1024 neurons (Leaky ReLU)  
  - 512 neurons (Leaky ReLU)  
  - 256 neurons (Leaky ReLU)  
- **Output**: Probability score (Sigmoid activation)  

## Training Details  
- **Epochs**: 100  
- **Batch Size**: 64  
- **Optimizer**: Adam (learning rate = 0.0002)  
- **Loss Function**: Binary Cross-Entropy (BCELoss)  

## Experimental Results  
| Model  | Final Discriminator Loss | Final Generator Loss |  
|--------|-------------------------|----------------------|  
| Model 1 | 1.0374 | 1.3149 |  
| Model 2 | 0.8044 | 1.6325 |  

Observations:  
- **Model 2** showed better stability due to **batch normalization** and **Leaky ReLU**.  
- Generated images improved significantly over training epochs.  

## Installation & Setup  
1. Clone the repository:  
   ```sh
   git clone https://github.com/your-repo/GAN-MNIST.git  
   cd GAN-MNIST  
   ```  
2. Install dependencies:  
   ```sh
   pip install torch torchvision matplotlib numpy  
   ```  
3. Run the training script:  
   ```sh
   python train.py  
   ```  

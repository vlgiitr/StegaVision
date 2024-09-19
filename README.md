# StegaVision: Enhancing Steganography with Attention Mechanism
----

## 🚀 Project Overview

In recent years, deep learning has shown significant potential in the field of image steganography, which involves hiding information within images in a way that is imperceptible to the human eye. Despite the progress made, many existing deep learning-based steganography methods still have limitations that need to be addressed.
Our study, StegaVision, aims to enhance image steganography by integrating attention mechanisms into an autoencoder-based model. Our approach focuses on dynamically adjusting the importance of different parts of the image through attention mechanisms. This helps in better embedding the hidden information while maintaining the image's visual quality.
We specifically explore two types of attention mechanisms—Channel Attention and Spatial Attention—and test their effectiveness on an autoencoder model.

### 🔍 Baseline Model

The foundation of our project is built upon a well-established autoencoder architecture, which serves as the baseline model. This section provides an overview of the baseline, highlighting its key components and the motivation behind its use.

### (A) Architecture

The baseline model utilizes a traditional autoencoder architecture, consisting of two main components:

- **Encoder**: This part of the model compresses the input data into a lower-dimensional latent space by capturing the most critical features.
- **Decoder**: It reconstructs the data from the latent representation, aiming to generate outputs that closely resemble the original inputs.

The autoencoder's design is particularly suited for tasks involving data compression, feature extraction, and reconstruction, making it an ideal choice for our steganography application.

### (B) Motivation

The choice of this autoencoder architecture as our baseline was driven by its effectiveness in preserving the essential features of input data while enabling efficient reconstruction. This architecture provides a robust and straightforward foundation, against which we can compare the enhancements introduced by incorporating attention mechanisms.

- Integration of Channel and Spatial Attention mechanisms
- Autoencoder-based architecture for efficient encoding and decoding
- Dynamic adjustment of image feature importance
- Improved embedding quality and visual fidelity

##  Motivation

The challenges in image steganography, such as optimizing the quality of stego images and ensuring precise message extraction, highlight the need for more advanced techniques that can overcome the limitations of traditional models. In response to these challenges, our project, **StegaVision** ,explores the integration of attention mechanisms into the steganographic process. 

Attention mechanisms, widely successful in other fields like natural language processing and computer vision, offer a promising approach to enhance steganography by dynamically focusing on the most significant features of images. This capability is crucial for improving the embedding process, making hidden messages more secure, and maintaining the integrity of the cover images.

By integrating Channel and Spatial Attention mechanisms into an autoencoder-based model, we aim to explore how these techniques can enhance the embedding process, improve the quality of stego images, and increase the precision of message extraction.
Ultimately, our goal is to demonstrate that attention mechanisms can address the persistent challenges in deep learning-based steganography, leading to more effective and undetectable methods for covert communication.


## 🛠️ Methodology

In this project, we systematically explored the impact of integrating attention mechanisms into an autoencoder-based steganography model. Our approach involved testing four distinct configurations to evaluate the effectiveness of both Channel and Spatial Attention mechanisms, individually and in combination.

### 1. Model Configurations

We tested the following five cases to understand the effects of different attention mechanisms:

- **Channel Attention Only**: In this configuration, we implemented only the Channel Attention mechanism in the autoencoder model. This approach focused on enhancing the model's ability to selectively emphasize important features across different channels of the image.

- **Spatial Attention Only**: Here, we applied only the Spatial Attention mechanism, allowing the model to concentrate on significant spatial features within the image, thereby improving the localization of embedded information.

- **Channel Followed by Spatial Attention**: In this case, we first applied Channel Attention to highlight key features across channels and then applied Spatial Attention to refine the focus on specific areas within the image. This sequential approach aimed to combine the strengths of both mechanisms.

- **Spatial Followed by Channel Attention**: Conversely, we first implemented Spatial Attention to prioritize crucial regions within the image, followed by Channel Attention to fine-tune the emphasis on channel-specific features. This setup allowed us to assess the impact of the order in which the attention mechanisms were applied.

- **Channel and Spatial Attention Parallely**: In this setup, both Channel and Spatial Attention mechanisms were applied simultaneously. This approach allowed the model to independently enhance feature importance across channels and spatial regions, providing a comprehensive focus on both dimensions concurrently. The goal was to leverage the strengths of each attention mechanism without the constraints of sequential application.

### 2. Learning Rate Scheduling

To optimize the training process, we incorporated a learning rate scheduler. This dynamic learning rate adjustment helped us fine-tune the model's performance by reducing the learning rate as training progressed, allowing the model to converge more effectively and avoid overfitting.

### 3. Dataset: ImageNet100

For our experiments, we used the ImageNet100 dataset, a subset of the larger ImageNet dataset. This dataset was chosen for its diversity and complexity, making it an ideal candidate for evaluating the robustness of our steganography model.

We prepared the dataset by creating random pairs of images to be used as cover and secret images. This pairing ensured a wide variety of combinations, which is essential for testing the generalization ability of the model across different types of images.

### 4. Training and Evaluation

Each model configuration was trained on the prepared dataset, with performance evaluated based on the quality of the generated stego images and the accuracy of message extraction. 

We evaluate our models using:
- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

## 📊 Results

The results, summarized in the following Table , show that using both attention mechanisms in parallel provides the best overall performance, striking an optimal balance between image quality and embedding accuracy.

| **Model**                  | **PSNR Cover** | **SSIM Cover** | **PSNR Secret** | **SSIM Secret** | **MSE Loss Cover** | **MSE Loss Secret** |
|----------------------------|----------------|----------------|-----------------|-----------------|--------------------|---------------------|
| **Baseline**                | 10.658         | 0.831          | 10.276          | 0.796           | 0.1620             | 0.1701              |
| **Channel Only**            | 10.666         | 0.831          | 10.289          | 0.797           | 0.1619             | 0.1699              |
| **Spatial Only**            | 10.734         | 0.835          | 10.266          | 0.799           | 0.1612             | 0.1703              |
| **Channel-Spatial Parallel**| 10.672         | 0.832          | 10.431          | 0.808           | 0.1616             | 0.1684              |
| **Channel then Spatial**    | 10.504         | 0.815          | 10.090          | 0.779           | 0.1640             | 0.1725              |
| **Spatial then Channel**    | 10.614         | 0.827          | 10.172          | 0.787           | 0.1627             | 0.1715              |

The performance metrics were evaluated using Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Mean Squared Error (MSE) for both the cover and secret images. The model with parallel channel and spatial attention outperformed the baseline and other configurations.

The results demonstrate that attention mechanisms are a valuable addition to deep learning-based image steganography. They significantly improve feature representation, helping to better hide secret images while maintaining cover image quality. This study lays the groundwork for further advancements in this field, providing a promising direction for future research.


## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/vlgiitr/StegaVision.git

# Install dependencies
pip install -r requirements.txt

# Run the training script
python main.py
```

----

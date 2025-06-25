# Vision-Language Model Comparison for Image Captioning

This project provides a framework for training and evaluating three popular vision-language models for the task of image captioning: BLIP, GIT, and ViLT. The models are trained on subsets of the COCO 2015 dataset, and their performance is evaluated using standard image captioning metrics.

## Table of Contents

- [Project Overview](#project-overview)
- [Models](#models)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Running the Training](#running-the-training)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Overview

The primary goal of this project is to compare the performance of BLIP, GIT, and ViLT on the image captioning task. The training script (`main.py`) is designed to:

- Load subsets of the COCO 2015 dataset.
- Fine-tune each of the three models on these subsets.
- Evaluate the trained models against a test set.
- Log training progress and evaluation metrics.
- Save the trained model checkpoints and evaluation results.

## Models

The following pre-trained models from the Hugging Face library are used as a starting point for fine-tuning:

- **BLIP (Bootstrapping Language-Image Pre-training)**: `Salesforce/blip-image-captioning-base`
- **GIT (GenerativeImage2Text)**: `microsoft/git-base`
- **ViLT (Vision-and-Language Transformer)**: `dandelin/vilt-b32-mlm`

## Dataset

The project uses the [COCO (Common Objects in Context) 2015 dataset](https://cocodataset.org/#home). To make the training process more manageable, the script is configured to use smaller subsets of the training data (1%, 5%, and 10%).

You will need to download the COCO 2015 images and annotations and structure them as described in the [Project Structure](#project-structure) section.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch
- CUDA (recommended for GPU acceleration)

### Installation

1. **Clone the repository:**

   ```bash
   git clone [https://github.com/your-username/vision-language-model-comparison.git](https://github.com/your-username/vision-language-model-comparison.git)
   cd vision-language-model-comparison

2. **Install the required Python packages:**

```bash
pip install -r requirements.txt


3. **Running the Training:****Clone the repository:**

```bash
python main.py

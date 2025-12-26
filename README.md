# 🖼️ Semantic GNN-Based Image Inpainting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/github/license/your-username/semantic-gnn-inpainting)](LICENSE)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-Track_Experiments-ffbe00?logo=weightsandbiases)](https://wandb.ai)

A **multi-scale Graph Neural Network (GNN)** for high-fidelity image inpainting that leverages semantic feature similarity to reconstruct missing regions with contextual awareness.

 

---

## 🌟 Key Features

- **Semantic Graph Construction**: Dynamically builds graphs using CNN-extracted features with dot-product similarity
- **Multi-Scale Reasoning**: Combines local (within-image) and global (cross-batch) contextual information
- **Hybrid Architecture**: Integrates message-passing GNN layers with attention-based GAT layers
- **Perceptual Optimization**: Uses LPIPS (Learned Perceptual Image Patch Similarity) for photorealistic results
- **Flexible Masking**: Supports rectangles, circles, and brush-stroke masks with adaptive coverage
- **Production-Ready**: Includes NaN protection, gradient clipping, and deterministic validation

---

##  Installation

###  Clone the repository
```bash
git clone https://github.com/your-username/semantic-gnn-inpainting.git
cd semantic-gnn-inpainting



```
 ##  Create a virtual environment 

```bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
##  Install dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
lpips>=0.1.4
wandb>=0.15.0
Pillow>=9.5.0
matplotlib>=3.7.0
```



##  Visualize results (after training)
Add this to the end of train.py or create inference.py:

```bash
from train import InpaintingModel, predict_and_visualize
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InpaintingModel(device, hidden_dim=128, feat_dim=84).to(device)
checkpoint = torch.load('best_semantic_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Generate and display results
predict_and_visualize(
    model, 
    image_path="train/sample_image.jpg", 
    device=device,
    img_size=128,
    mask_ratio=0.4
)
```
##   Repository Structure
```bash
semantic-gnn-inpainting/
├── train/                  # Training images (add your data here)
├── train.py                # Main training script
├── best_semantic_model.pth # Saved model checkpoint (auto-generated)
├── README.md               # This file
└── .gitignore              # Recommended: Add wandb/, __pycache__, etc.
```

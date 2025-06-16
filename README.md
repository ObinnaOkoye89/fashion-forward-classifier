# Fashion Forward Garment Classifier 👗👖👟

Fashion Forward is an AI-powered e-commerce clothing retailer. This project demonstrates how Convolutional Neural Networks (CNNs) can automate product tagging by classifying clothing images into categories like shirts, trousers, and shoes using the FashionMNIST dataset.

## 🧠 Model Overview
- Architecture: Simple CNN with 2 Conv layers + Fully connected layers
- Dataset: FashionMNIST (10 clothing categories)
- Task: Multi-class classification
- Framework: PyTorch + TorchMetrics

## 📊 Evaluation Metrics
- Accuracy
- Per-class Precision
- Per-class Recall

## 🧪 Output
After training for 1 epoch, the model:
- Stores predictions in a `predictions` list
- Computes overall `accuracy`
- Stores per-class `precision` and `recall` in lists

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run training and evaluation
python garment_classifier.py

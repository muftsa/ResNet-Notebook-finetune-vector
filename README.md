# ResNet-Notebook-finetune-vector

To evaluate the effectiveness of continued ResNet fine-tuning over time, you can use a vector database to store and retrieve feature embeddings. This allows you to track how well the model separates classes by comparing embeddings from different training stages.

### Approach:
1. **Extract Feature Embeddings**: Use the trained ResNet model to extract embeddings from the validation set.
2. **Store Embeddings in a Vector Database**: Store these embeddings with class labels in a vector database (e.g., `FAISS`, `Pinecone`, or `Weaviate`).
3. **Evaluate Effectiveness**: At each checkpoint, compare new embeddings to prior embeddings to track model performance and evolution over time.

For simplicity, we'll use `FAISS` here, a popular open-source vector database suitable for Jupyter notebooks.

### Requirements:
1. Install necessary libraries:
   ```bash
   pip install torch torchvision faiss-cpu
   ```

### Jupyter Notebook Code

This notebook will:
1. Train the ResNet model.
2. Extract embeddings at checkpoints.
3. Store and retrieve embeddings in FAISS.
4. Compute evaluation metrics to track the model's effectiveness.

```python
# Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import faiss
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define paths for training and validation data
train_dir = 'data/train'
val_dir = 'data/val'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load validation dataset
val_dataset = ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet and modify the final layer
model = models.resnet18(pretrained=True)
num_classes = len(val_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Set up FAISS vector store for tracking embeddings over time
dimension = model.fc.in_features  # Embedding dimension from ResNet
index = faiss.IndexFlatL2(dimension)

# Define helper functions
def extract_embeddings(model, loader):
    """
    Extract embeddings from the model for the given DataLoader.
    """
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(embeddings), np.hstack(labels)

def store_embeddings_in_faiss(index, embeddings, labels):
    """
    Store embeddings in the FAISS index with labels for comparison.
    """
    index.add(embeddings)
    print(f"Stored {len(labels)} embeddings in FAISS index.")

def evaluate_embeddings(index, embeddings, labels):
    """
    Evaluate model effectiveness by checking nearest neighbors in FAISS.
    """
    D, I = index.search(embeddings, k=1)  # Search for the closest neighbor
    matches = (I.flatten() == labels)
    accuracy = matches.sum() / len(labels)
    return accuracy

# Training loop with periodic evaluation
num_epochs = 10
checkpoint_interval = 2  # Evaluate every 2 epochs

for epoch in range(num_epochs):
    # Here we would have our model training code (not shown for brevity)

    # Periodic evaluation
    if (epoch + 1) % checkpoint_interval == 0:
        print(f"Epoch {epoch+1} - Extracting and storing embeddings")
        
        # Extract embeddings from validation set
        val_embeddings, val_labels = extract_embeddings(model, val_loader)
        
        # Store embeddings in FAISS for current checkpoint
        store_embeddings_in_faiss(index, val_embeddings, val_labels)
        
        # Evaluate model's embedding effectiveness
        accuracy = evaluate_embeddings(index, val_embeddings, val_labels)
        print(f"Validation Accuracy at Epoch {epoch+1}: {accuracy:.4f}")

print("Training and evaluation complete.")
```

### Explanation

1. **Setup and Data Loading**:
   - Load and preprocess images.
   - Load a pre-trained ResNet model and modify it for your number of classes.

2. **FAISS Vector Database**:
   - Use FAISS’s `IndexFlatL2` index to store embeddings.
   - The `dimension` is determined by the model’s embedding layer (`model.fc.in_features`).

3. **Embedding Extraction**:
   - `extract_embeddings`: Runs inference on the model’s last layer to get feature embeddings.
   - `store_embeddings_in_faiss`: Stores embeddings in FAISS at each checkpoint.

4. **Evaluate Effectiveness**:
   - `evaluate_embeddings`: Uses FAISS to find the nearest neighbor of each embedding and evaluates how many are correctly matched to their class label.
   - Track model improvement with checkpointed evaluation.

5. **Periodic Evaluation**:
   - The notebook evaluates the model at each checkpoint interval, storing and comparing embeddings to track performance over time.

This setup will allow you to store and evaluate embeddings over time to monitor the model’s learning progress and effectiveness on the validation set.

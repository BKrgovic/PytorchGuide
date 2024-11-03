from neuralNet import NeuralNetwork, training_data
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Load the model and state dictionary
model = NeuralNetwork()
model.load_state_dict(torch.load(r"C:\Users\bkrgo\OneDrive\Desktop\Pytorch\statedict", weights_only=True))
model.eval()

# Labels mapping
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# Plotting the images with predictions and true labels
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, true_label = training_data[sample_idx]

    # Model prediction
    img = img.unsqueeze(0)  # Add batch dimension if needed
    logits = model(img)
    pred_probab = nn.Softmax(dim=1)(logits)
    predicted_label = pred_probab.argmax(1).item()  # Convert to int

    # Plotting with true and predicted labels
    figure.add_subplot(rows, cols, i)
    plt.title(f"True: {labels_map[true_label]} \nPred: {labels_map[predicted_label]}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()

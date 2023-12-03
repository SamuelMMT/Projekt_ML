import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Path to your dataset
dataset_path = os.getcwd() + '/traffic_light_data'

transform = transforms.Compose([
    transforms.Resize((64, 64)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# The data:
train_ds = torchvision.datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
val_ds = torchvision.datasets.ImageFolder(root=f"{dataset_path}/val", transform=transform)
batch_size = 64
train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True)
val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=True)

# The model:
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  #3 input layers, 16 outputs
            nn.BatchNorm2d(16),             #Does it make sense for this batch size? Stabilization through adjusting and scaling the activation
            nn.ReLU(inplace=True),          #Activation (Rectified Linear Unit) 
            nn.MaxPool2d(kernel_size=2, stride=2), #To summarize features and make computation faster 
                                                    #(higher-level features don't require a high spatial resolution)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # 4 output classes: red, yellow, green, back
        )

    def forward(self, X0):
        # Forward pass method here
        X1 = self.cnn_layers(X0)
        X2 = X1.view(X1.size(0), -1)
        X3 = self.linear_layers(X2)
        return X3

    def loss(self, Y_true, Y_pred):
        return nn.CrossEntropyLoss()(Y_pred, Y_true)

def train_step(X, Y_true, mdl, opt):
    # Your current train_step function here
    Y_pred = mdl(X)
    L = mdl.loss(Y_true, Y_pred)  
    L.backward() 
    opt.step()
    opt.zero_grad() 
    return L.detach().numpy()

def train(train_dl, val_dl, mdl, alpha, max_epochs, target_val_loss = None, min_delta = 0.001):
    opt = torch.optim.Adam(mdl.parameters(), lr=alpha)
    hist = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(max_epochs):
        mdl.train()
        for step, (X, Y_true) in enumerate(train_dl):
            L = train_step(X, Y_true, mdl, opt)
            hist['train_loss'].append(L)

        mdl.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, Y_val_true in val_dl:
                val_loss = mdl.loss(Y_val_true, mdl(X_val)).item()
                val_losses.append(val_loss)

        hist['val_loss'].extend(val_losses)
        
        avg_train_loss = sum(hist['train_loss'][-len(train_dl):]) / len(train_dl)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        print(f'Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Check for improvement in validation loss
        if target_val_loss is not None and best_val_loss - avg_val_loss > min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check stopping conditions
        if epochs_without_improvement >= 3:  # Adjust as needed
            print("Stopping early due to lack of improvement.")
            break
        
        
    return hist

mdl = Model()
max_epochs = 20
hist = train(train_dl, val_dl, mdl, alpha=0.00001, max_epochs=max_epochs, target_val_loss=0.1)

def plot_losses(train_loss, val_loss, num_train_batches, num_val_batches):
    # Create x-axis values for the training and validation losses
    train_x = np.arange(len(train_loss)) / num_train_batches
    val_x = np.arange(len(val_loss)) * num_val_batches / num_train_batches

    plt.figure()
    plt.plot(train_x, train_loss, label='Training Loss')
    plt.plot(val_x, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Example usage
num_train_batches = len(train_dl)
num_val_batches = len(val_dl)
plot_losses(hist['train_loss'], hist['val_loss'], num_train_batches, num_val_batches)
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:19:37 2024

@author: Praveen Jawaharlal Ayyanathan
Training and testing data are generated using https://github.com/jp-praveen/Obstacle_Avoidance_Convex_Optimization
Original paper link: https://arc.aiaa.org/doi/abs/10.2514/6.2025-0565
"""

# Import important libraries
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import random

# Setting random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Load training data for NN input
train_data = sio.loadmat('inputs_train_1obs_2obs.mat')
inputs_train = train_data['inputs_train_1obs_2obs']
outputs_train_data = sio.loadmat('outputs_train_1obs_2obs.mat')
outputs_train = outputs_train_data['outputs_train_1obs_2obs']

# Load testing data for NN output
test_data = sio.loadmat('inputs_test_1obs_2obs.mat')
inputs_test = test_data['inputs_test_1obs_2obs']
outputs_test_data = sio.loadmat('outputs_test_1obs_2obs.mat')
outputs_test = outputs_test_data['outputs_test_1obs_2obs']

# NO scalers applied currently
# If needed edit the next two lines
inputs_train_adjusted = inputs_train
inputs_test_adjusted = inputs_test

# Split the training data into training and validation sets
inputs_train_final, inputs_val, outputs_train_final, outputs_val = train_test_split(
    inputs_train_adjusted, outputs_train, test_size=0.1, random_state=42)

# Converting to PyTorch tensors
inputs_train_tensor = torch.from_numpy(inputs_train_final).float()
outputs_train_tensor = torch.from_numpy(outputs_train_final).float()
inputs_val_tensor = torch.from_numpy(inputs_val).float()
outputs_val_tensor = torch.from_numpy(outputs_val).float()
test_inputs_tensor = torch.from_numpy(inputs_test_adjusted).float()
test_outputs_tensor = torch.from_numpy(outputs_test).float()

# Loading the best hyperparameters given by Optuna
best_hyperparameters = joblib.load('best_hyperparameters_min_1obs_2obs.pkl')
print("Best Hyperparameters:")
for key, value in best_hyperparameters.items():
    print(f"  {key}: {value}")

# Extracting the architectutal and optimization hyperparameters
best_n_layers = best_hyperparameters['n_layers']
best_hidden_size = best_hyperparameters['hidden_size']
best_activation_name = best_hyperparameters['activation']
best_batch_size = best_hyperparameters['batch_size']
best_learning_rate = best_hyperparameters['learning_rate']
best_dropout_rate = best_hyperparameters['dropout_rate']
best_weight_decay = best_hyperparameters['weight_decay']
best_beta1 = best_hyperparameters['beta1']
best_beta2 = best_hyperparameters['beta2']
best_patience = best_hyperparameters['patience']
best_use_scheduler = best_hyperparameters['use_scheduler']
if best_use_scheduler:
    best_scheduler_factor = best_hyperparameters['scheduler_factor']
    best_scheduler_patience = best_hyperparameters['scheduler_patience']

# Activation function
activation_map = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
    'SELU': nn.SELU,
}
activation_class = activation_map[best_activation_name]

# Build the neural network based on hyperparameters
layers = []
in_features = inputs_train_adjusted.shape[1]
for i in range(best_n_layers):
    layers.append(nn.Linear(in_features, best_hidden_size))
    if best_activation_name == 'SELU':
        layers.append(nn.SELU())
        if best_dropout_rate > 0.0:
            layers.append(nn.AlphaDropout(best_dropout_rate))
    else:
        layers.append(activation_class())
        if best_dropout_rate > 0.0:
            layers.append(nn.Dropout(best_dropout_rate))
    in_features = best_hidden_size
layers.append(nn.Linear(best_hidden_size, outputs_train.shape[1]))  # Output layer

# Create the model
model = nn.Sequential(*layers)

# Loss function - Mean Square Erro
criterion = nn.MSELoss()

# Optimizer - Adam
optimizer = optim.Adam(
    model.parameters(),
    lr=best_learning_rate,
    weight_decay=best_weight_decay,
    betas=(best_beta1, best_beta2)
)

# Learning rate scheduler
if best_use_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=best_scheduler_factor, patience=best_scheduler_patience
    )

# Dataset and dataloader for training data
train_dataset = torch.utils.data.TensorDataset(inputs_train_tensor, outputs_train_tensor)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)

# Data set and dataloader for validation data
val_dataset = torch.utils.data.TensorDataset(inputs_val_tensor, outputs_val_tensor)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)

# Number of epochs
num_epochs = 500

# Initialize lists to store loss values
train_loss_values = []
val_loss_values = []

# Early stopping parameters
patience = best_patience
min_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for batch_inputs, batch_outputs in train_dataloader:
        optimizer.zero_grad()
        predictions = model(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * batch_inputs.size(0)
    epoch_train_loss /= len(train_dataloader.dataset)
    train_loss_values.append(epoch_train_loss)

    # Compute validation loss
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_outputs in val_dataloader:
            val_predictions = model(val_inputs)
            val_loss = criterion(val_predictions, val_outputs)
            epoch_val_loss += val_loss.item() * val_inputs.size(0)
    epoch_val_loss /= len(val_dataloader.dataset)
    val_loss_values.append(epoch_val_loss)

    # Early stopping 
    if epoch_val_loss < min_val_loss:
        min_val_loss = epoch_val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1

    # Learning rate scheduler step
    if best_use_scheduler:
        scheduler.step(epoch_val_loss)

    if epochs_no_improve >= patience:
        print('Early stopping!')
        break

    print('Epoch [{}/{}], Training Loss: {:.5f}, Validation Loss: {:.5f}'.format(
        epoch+1, num_epochs, epoch_train_loss, epoch_val_loss))

# Load the best model parameters
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save the model
torch.save(model.state_dict(), 'trained_model_min_1obs_2obs.pth')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = model(test_inputs_tensor)
    test_loss = criterion(predictions, test_outputs_tensor)
    print('Test Loss: {:.5f}'.format(test_loss.item()))

# Plot the training and validation loss curves
plt.figure(figsize=(10, 6))
epochs_range = range(1, len(train_loss_values) + 1)
plt.plot(epochs_range, train_loss_values, label='Training Loss')
plt.plot(epochs_range, val_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Save training and validation losses
sio.savemat('NN1_min_2obs_ofit_loss_data.mat', {
    'train_loss_values': train_loss_values,
    'val_loss_values': val_loss_values,
    'test_loss':test_loss.item()
})

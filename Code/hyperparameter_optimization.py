# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:47:04 2024

@author: Praveen Jawaharlal Ayyanathan
Training and testing data are generated using https://github.com/jp-praveen/Obstacle_Avoidance_Convex_Optimization
Original paper link: https://arc.aiaa.org/doi/abs/10.2514/6.2025-0565
This code uses optuna to optimize the hyperparamters of a Neural Network that can
generate collision free trajectories in the presence of static obstacles

"""

# Import important libraries
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split
import joblib
from optuna.visualization import plot_param_importances
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load training data for NN input
train_data = sio.loadmat('inputs_train_1obs_2obs.mat')

# Load training data for NN output
outputs_train = sio.loadmat('outputs_train_1obs_2obs.mat')

inputs_all = train_data['inputs_train_1obs_2obs']
outputs_all = outputs_train['outputs_train_1obs_2obs']

# Generating validation data from training data
inputs_train_final, inputs_val, outputs_train_final, outputs_val = train_test_split(
    inputs_all, outputs_all, test_size=0.2, random_state=42)

# NO scalers applied currently
# If needed edit the next four lines
inputs_train_scaled = inputs_train_final
outputs_train_scaled = outputs_train_final
inputs_val_scaled = inputs_val
outputs_val_scaled = outputs_val

# Starting Optuna trials
def objective(trial):

    # Defining the range of hyperparameters
    n_layers = trial.suggest_int('n_layers', 2, 5)
    hidden_size = trial.suggest_int('hidden_size', 16, 1024, log=True)
    activation_name = trial.suggest_categorical(
        'activation', ['ReLU', 'Tanh', 'LeakyReLU', 'ELU', 'SELU']
    )
    batch_size = trial.suggest_int('batch_size', 16, 512, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    beta1 = trial.suggest_float('beta1', 0.85, 0.99)
    beta2 = trial.suggest_float('beta2', 0.9, 0.9999)
    patience = trial.suggest_int('patience', 5, 20)
    use_scheduler = trial.suggest_categorical('use_scheduler', [True, False])
    if use_scheduler:
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.9)
        scheduler_patience = trial.suggest_int('scheduler_patience', 2, 10)

    # Define the activation function
    activation_map = {
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
        'LeakyReLU': nn.LeakyReLU,
        'ELU': nn.ELU,
        'SELU': nn.SELU,
    }
    activation_class = activation_map[activation_name]
    
    # Build the neural network based on hyperparameters
    layers = []

    # Input layer size
    in_features = inputs_train_scaled.shape[1]

    # Form the Deep Neural Network layers
    for i in range(n_layers):
        layers.append(nn.Linear(in_features, hidden_size))
        if activation_name == 'SELU':
            layers.append(nn.SELU())
            if dropout_rate > 0.0:
                layers.append(nn.AlphaDropout(dropout_rate))
        else:
            layers.append(activation_class())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
        in_features = hidden_size
    layers.append(nn.Linear(hidden_size, outputs_train_scaled.shape[1]))  # Output layer

    # Create the model
    model = nn.Sequential(*layers)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )

    # Learning rate scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )

    # Convert data to PyTorch tensors
    inputs_train_tensor = torch.from_numpy(inputs_train_scaled).float()
    outputs_train_tensor = torch.from_numpy(outputs_train_scaled).float()
    inputs_val_tensor = torch.from_numpy(inputs_val_scaled).float()
    outputs_val_tensor = torch.from_numpy(outputs_val_scaled).float()

    # Create DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(inputs_train_tensor, outputs_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Training loop with monitoring of training and validation loss
    num_epochs = 150
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None 
    best_epoch = 0  

    # Start Training
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_inputs, batch_outputs in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_inputs.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_predictions = model(inputs_val_tensor)
            val_loss = criterion(val_predictions, outputs_val_tensor).item()
        val_losses.append(val_loss)

        # Printing the training and validatation losses over certain intervals
        #if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == num_epochs:
           # print(
            #    f'Epoch [{epoch+1}/{num_epochs}], '
             #   f'Training Loss: {epoch_train_loss:.5f}, '
              #  f'Validation Loss: {val_loss:.5f}'
            #)

        # Check for improvement. Useful to prevent overfitting
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save the best model parameters
            best_epoch = epoch  # Save the epoch number
        else:
            epochs_no_improve += 1

        # Learning rate scheduler step
        if use_scheduler:
            scheduler.step(val_loss)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        # Report intermediate objective value to Optuna
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Load the best model parameters after training
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Store the losses 
    trial.set_user_attr('train_losses', train_losses)
    trial.set_user_attr('val_losses', val_losses)
    trial.set_user_attr('best_epoch', best_epoch)

    # Compute overfitting measure using the best epoch
    overfitting_penalty = min_val_loss - train_losses[best_epoch]
    if overfitting_penalty < 0:
        overfitting_penalty = 0  # No penalty if validation loss is lower than training loss

    # Overfitting penalty in objective value
    alpha = 0.3  # Weight for overfitting penalty
    objective_value = min_val_loss + alpha * overfitting_penalty

    return objective_value

# Study to minimize the objective
study = optuna.create_study(direction='minimize')

# Start the optimization with a limit on the trials or timeout
study.optimize(objective, n_trials=300, timeout=12*3600)  

print('Number of finished trials: ', len(study.trials))
print('Best trial:')
trial = study.best_trial

print('  Value (Objective Value): {:.5f}'.format(trial.value))
print('  Params:')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Save the best hyperparameters
joblib.dump(trial.params, 'best_hyperparameters_min_1obs_2obs.pkl')

# Visualization
fig = plot_param_importances(study)
plt.show()

# Retrieve the training and validation losses from the best trial
best_trial = study.best_trial
train_losses = best_trial.user_attrs['train_losses']
val_losses = best_trial.user_attrs['val_losses']
best_epoch = best_trial.user_attrs['best_epoch']

# Plot the training and validation losses
plt.figure(figsize=(10, 6))
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
plt.axvline(x=best_epoch+1, color='g', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses Over Epochs for Best Trial')
plt.legend()
plt.show()

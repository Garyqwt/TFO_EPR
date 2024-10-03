import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from loss import weighted_mse_loss, weighted_l1_loss

def train(model, X_train, y_train, device, sample_cnt=None, batch_size=32, X_test=None, y_test=None, epochs=100, val_patience=15,
          lr=1e-4, weight_decay=1e-4, momentum=0.98, nesterov=True):
    train_dataset = TensorDataset(X_train, y_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = epochs
    validate_every = 1
    best_val_loss = float('inf')  # Initialize with a large value
    early_stop_counter = 0
    best_model_state = None

    train_losses = []
    test_losses = []
    train_mae = []
    test_mae = []

    # Training loop
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        running_mae = 0.0
        
        for inputs, info in train_data_loader: 
            inputs, info = inputs.to(device), info.to(device)
            labels = info[:,0]
            rounds = info[:,1]
            weights = [np.float32(sum(sample_cnt) / sample_cnt[int(x)]) for x in rounds] if sample_cnt is not None else None
            optimizer.zero_grad()  # Zero the gradient buffers
            labels = labels.view(-1)
            reg_out = model(inputs) # Forward pass
            reg_out = reg_out.view(-1)
            weights = torch.Tensor(weights).to(device) if weights is not None else None
            reg_loss = weighted_mse_loss(reg_out, labels, weights=weights)  # Calculate weighted MSE
            mae_loss = weighted_l1_loss(reg_out, labels, weights=weights)  # Calculate weighted MAE
            
            
            loss = reg_loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()
            running_mae += mae_loss.item()
            
        # Print the average loss and MAE for this epoch
        avg_loss = running_loss / len(train_data_loader)
        avg_mae = running_mae / len(train_data_loader)
        train_losses.append(avg_loss)
        train_mae.append(avg_mae)
        
        if X_test is not None and y_test is not None:
            if (epoch + 1) % validate_every == 0:
                # Validate the model
                avg_val_loss, avg_val_mae = validation(model, X_test, y_test, device, verbose=0)
                test_losses.append(avg_val_loss)
                test_mae.append(avg_val_mae)
                print(f'Epoch [{epoch + 1}/{num_epochs}] - train loss: {avg_loss:.4e} - train MAE: {avg_mae:.4e} - test loss: {avg_val_loss:.4e} - test MAE: {avg_val_mae:.4e}', flush=True)

                # Check for improvement in validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = model.state_dict()
                    early_stop_counter = 0  
                else:
                    early_stop_counter += 1

                # Early stopping condition
                if early_stop_counter >= val_patience:
                    print(f'Early stopping at epoch {epoch + 1} with patience {val_patience} epochs.', flush=True)
                    break
        else:
            print(f'Epoch [{epoch + 1}/{num_epochs}] - train loss: {avg_loss:.4e} - train MAE: {avg_mae:.4e}', flush=True)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    loss_history = {
        'train_loss': train_losses,
        'train_mae': train_mae,
        'test_loss': test_losses,
        'test_mae': test_mae
    }

    return model, min(train_losses), min(train_mae), min(test_losses), min(test_mae), loss_history


def validation(model, X_test, y_test, device, verbose=0):
    # Validation (testing) loop
    model.eval()  # Set the model to evaluation mode
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    test_dataset = TensorDataset(X_test, y_test)
    test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    criterion_reg = nn.MSELoss()
    MAE_metric = nn.L1Loss()

    test_loss = 0.0
    # test_reg_loss = 0.0
    test_mae = 0.0

    with torch.no_grad():
        for inputs, info in test_data_loader:
            inputs, info = inputs.to(device), info.to(device)
            labels = info[:,0]
            rounds = info[:,1]
            labels = labels.view(-1) 
            reg_out = model(inputs)  # Forward pass
            reg_out = reg_out.view(-1)
            reg_loss = criterion_reg(reg_out, labels)
            mae_loss = MAE_metric(reg_out, labels)
            loss = reg_loss #my_loss_fn(reg_loss, cla_loss)

            test_loss += loss.item()
            test_mae += mae_loss.item()

    # Print the average test loss and MAE
    avg_test_loss = test_loss / len(test_data_loader)
    avg_test_mae = test_mae / len(test_data_loader)
    if verbose == 1:
        print(f'Test Loss: {avg_test_loss:.4f} - Test MAE: {avg_test_mae:.4f}')
    return avg_test_loss, avg_test_mae


def inference(model, X, device):
    model.eval()  # Set the model to evaluation mode
    X = torch.Tensor(X)
    test_dataset = TensorDataset(X)
    test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    predictions = []

    with torch.no_grad():
        for inputs in test_data_loader:
            inputs = inputs[0].to(device)
            reg_out = model(inputs)  # Forward pass
            reg_out = reg_out.view(-1)
            predictions.append(reg_out.cpu().numpy())

    # Concatenate predictions from all batches
    predictions = np.concatenate(predictions)

    return predictions
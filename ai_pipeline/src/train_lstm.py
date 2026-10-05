import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

from lstm_preprocessing import load_and_preprocess_lstm_data

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'bridge_digital_twin_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'lstm')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'notebooks', 'lstm')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# Hyperparameters
SEQ_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 10

class BridgeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=1, dropout=0.2):
        super(BridgeLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def plot_history(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.title('LSTM Training History (PyTorch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(12, 6))
    subset = min(500, len(y_true))
    plt.plot(y_true[-subset:], label='Actual SHI', color='blue', alpha=0.7)
    plt.plot(y_pred[-subset:], label='Predicted SHI', color='red', alpha=0.7, linestyle='--')
    plt.title(f'Actual vs Predicted SHI (Last {subset} steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Structural Health Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred.flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residuals Distribution (Actual - Predicted)')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    scaler_path = os.path.join(MODEL_DIR, 'lstm_scaler.joblib')
    X, y, feature_names = load_and_preprocess_lstm_data(DATASET_PATH, SEQ_LENGTH, scaler_path)
    
    # Save feature names
    features_path = os.path.join(MODEL_DIR, 'lstm_features.json')
    with open(features_path, 'w') as f:
        json.dump({'features': feature_names, 'seq_length': SEQ_LENGTH}, f)
        
    # Chronological Split (Train: 70%, Val: 10%, Test: 20%)
    train_end = int(len(X) * 0.7)
    val_end = int(len(X) * 0.8)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Convert to PyTorch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[2]
    model = BridgeLSTM(input_dim=input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training PyTorch LSTM model...")
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Load best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    # Save model
    model_checkpoint_path = os.path.join(MODEL_DIR, 'shi_lstm_model.pth')
    torch.save(model.state_dict(), model_checkpoint_path)
    
    print("\nEvaluating model on test set...")
    model.eval()
    test_preds = []
    test_actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            test_preds.extend(outputs.cpu().numpy())
            test_actuals.extend(batch_y.numpy())
            
    test_preds = np.array(test_preds).flatten()
    test_actuals = np.array(test_actuals).flatten()
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    mae = mean_absolute_error(test_actuals, test_preds)
    r2 = r2_score(test_actuals, test_preds)
    
    metrics_path = os.path.join(NOTEBOOK_DIR, 'lstm_performance_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== LSTM Regression Performance Metrics (PyTorch) ===\n")
        f.write(f"Target: Structural_Health_Index_SHI\n")
        f.write(f"Sequence Length: {SEQ_LENGTH}\n")
        f.write("-" * 40 + "\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"R2:   {r2:.4f}\n")
        
    print(f"Metrics saved to {metrics_path}")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # Plots
    plot_history(train_losses, val_losses, os.path.join(NOTEBOOK_DIR, 'lstm_training_history.png'))
    plot_predictions(test_actuals, test_preds, os.path.join(NOTEBOOK_DIR, 'lstm_actual_vs_predicted.png'))
    plot_residuals(test_actuals, test_preds, os.path.join(NOTEBOOK_DIR, 'lstm_residuals.png'))
    
    print("LSTM training pipeline completed successfully.")

if __name__ == "__main__":
    main()

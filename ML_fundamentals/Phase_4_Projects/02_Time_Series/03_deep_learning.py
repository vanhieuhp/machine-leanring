"""
Time Series Forecasting - Part 3: Deep Learning for Time Series
================================================================

This module covers:
- LSTM networks for time series
- Sequence preparation (sliding windows)
- PyTorch implementation
- Keras/TensorFlow implementation
- Multi-step forecasting

Based on: Financial Forecasting (Stock Prices)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================

print("=" * 70)
print("1. DATA PREPARATION FOR DEEP LEARNING")
print("=" * 70)

# Create sample data
np.random.seed(42)
n = 500

dates = pd.date_range(start='2022-01-01', periods=n, freq='B')
trend = np.linspace(100, 150, n)
seasonal = 10 * np.sin(np.linspace(0, 8 * np.pi, n))
noise = np.random.randn(n) * 3
prices = trend + seasonal + noise

df = pd.DataFrame({'Close': prices}, index=dates)

# Split data
train_size = int(len(df) * 0.8)
train_data = df['Close'][:train_size].values
test_data = df['Close'][train_size:].values

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# -------------------------------------------------------------------------
# 1.1 Scaling
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.1 Feature Scaling")
print("-" * 50)

# MinMaxScaler (recommended for neural networks)
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

print(f"Original range: [{train_data.min():.2f}, {train_data.max():.2f}]")
print(f"Scaled range: [{train_scaled.min():.2f}, {train_scaled.max():.2f}]")

# -------------------------------------------------------------------------
# 1.2 Create Sequences (Sliding Window)
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("1.2 Creating Sequences (Sliding Window)")
print("-" * 50)

def create_sequences(data, seq_length):
    """
    Create sequences for time series prediction.

    Parameters:
    -----------
    data : array-like
        Time series data
    seq_length : int
        Number of time steps to use for prediction

    Returns:
    --------
    X, y : arrays
        Input sequences and targets
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Sequence length (lookback window)
seq_length = 20

# Create training sequences
X_train, y_train = create_sequences(train_scaled, seq_length)

print(f"Sequence length (lookback): {seq_length}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Create test sequences (use last part of training data)
# Need to include some training data for initial sequence
full_data = np.concatenate([train_scaled, test_scaled])
X_test, y_test = create_sequences(full_data, seq_length)

# Only use test portion
X_test = X_test[len(train_scaled) - seq_length:]
y_test = y_test[len(train_scaled) - seq_length:]

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Reshape for neural networks [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"\nReshaped X_train: {X_train.shape}")
print(f"Reshaped X_test: {X_test.shape}")

# ============================================================================
# 2. PYTORCH LSTM
# ============================================================================

print("\n" + "=" * 70)
print("2. PYTORCH LSTM IMPLEMENTATION")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print("PyTorch version:", torch.__version__)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2.1 Define LSTM Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.1 Defining LSTM Model")
    print("-" * 50)

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            self.fc1 = nn.Linear(hidden_size, 32)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # LSTM
            lstm_out, _ = self.lstm(x)

            # Take the last time step
            out = lstm_out[:, -1, :]

            # Fully connected layers
            out = self.relu(self.fc1(out))
            out = self.dropout(out)
            out = self.fc2(out)

            return out

    # Instantiate model
    lstm_model = LSTMModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )

    print(f"Model architecture:")
    print(lstm_model)

    # Count parameters
    total_params = sum(p.numel() for p in lstm_model.parameters())
    trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # 2.2 Training
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.2 Training LSTM")
    print("-" * 50)

    # Prepare data
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    lstm_model.train()

    print(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")

    # -------------------------------------------------------------------------
    # 2.3 Evaluation
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("2.3 Evaluating LSTM")
    print("-" * 50)

    lstm_model.eval()
    with torch.no_grad():
        train_pred = lstm_model(X_train_tensor).cpu().numpy()
        test_pred = lstm_model(X_test_tensor).cpu().numpy()

    # Inverse transform predictions
    train_pred_inv = scaler.inverse_transform(train_pred).flatten()
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    test_pred_inv = scaler.inverse_transform(test_pred).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate metrics
    lstm_train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
    lstm_test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    lstm_test_mae = mean_absolute_error(y_test_inv, test_pred_inv)

    print(f"\nLSTM Results:")
    print(f"  Train RMSE: {lstm_train_rmse:.4f}")
    print(f"  Test RMSE: {lstm_test_rmse:.4f}")
    print(f"  Test MAE: {lstm_test_mae:.4f}")

except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    lstm_test_rmse = None

# ============================================================================
# 3. KERAS/TENSORFLOW LSTM
# ============================================================================

print("\n" + "=" * 70)
print("3. KERAS/TENSORFLOW LSTM IMPLEMENTATION")
print("=" * 70)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    print("TensorFlow version:", tf.__version__)

    # -------------------------------------------------------------------------
    # 3.1 Define Model
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.1 Defining Keras LSTM Model")
    print("-" * 50)

    keras_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    keras_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("Model summary:")
    keras_model.summary()

    # -------------------------------------------------------------------------
    # 3.2 Training
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.2 Training Keras LSTM")
    print("-" * 50)

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train
    history = keras_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # -------------------------------------------------------------------------
    # 3.3 Evaluation
    # -------------------------------------------------------------------------

    print("\n" + "-" * 50)
    print("3.3 Evaluating Keras LSTM")
    print("-" * 50)

    # Predictions
    train_pred = keras_model.predict(X_train, verbose=0)
    test_pred = keras_model.predict(X_test, verbose=0)

    # Inverse transform
    train_pred_inv = scaler.inverse_transform(train_pred).flatten()
    test_pred_inv = scaler.inverse_transform(test_pred).flatten()

    # Metrics
    keras_train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
    keras_test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv))
    keras_test_mae = mean_absolute_error(y_test_inv, test_pred_inv)

    print(f"\nKeras LSTM Results:")
    print(f"  Train RMSE: {keras_train_rmse:.4f}")
    print(f"  Test RMSE: {keras_test_rmse:.4f}")
    print(f"  Test MAE: {keras_test_mae:.4f}")

except ImportError:
    print("TensorFlow not installed. Install with: pip install tensorflow")
    keras_test_rmse = None

# ============================================================================
# 4. GRU (GATED RECURRENT UNIT)
# ============================================================================

print("\n" + "=" * 70)
print("4. GRU IMPLEMENTATION")
print("=" * 70)

try:
    from tensorflow.keras.layers import GRU

    print("\n" + "-" * 50)
    print("4.1 Defining GRU Model")
    print("-" * 50)

    gru_model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    gru_model.compile(optimizer='adam', loss='mse')

    print("GRU Model:")
    gru_model.summary()

    # Train
    print("\nTraining GRU...")
    gru_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate
    gru_pred = gru_model.predict(X_test, verbose=0)
    gru_pred_inv = scaler.inverse_transform(gru_pred).flatten()
    gru_test_rmse = np.sqrt(mean_squared_error(y_test_inv, gru_pred_inv))

    print(f"\nGRU Test RMSE: {gru_test_rmse:.4f}")

except ImportError:
    print("TensorFlow not required for this section")
    gru_test_rmse = None

# ============================================================================
# 5. MULTI-STEP FORECASTING
# ============================================================================

print("\n" + "=" * 70)
print("5. MULTI-STEP FORECASTING")
print("=" * 70)

# -------------------------------------------------------------------------
# 5.1 Direct Multi-Step
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.1 Direct Multi-Step Forecasting")
print("-" * 50)

# Forecast multiple steps ahead
n_steps = 10

# Use the last sequence from test data as starting point
last_sequence = X_test[-1].copy()

# Direct prediction: train a model to predict N steps at once
# For simplicity, we'll use iterative prediction

predictions = []
current_sequence = last_sequence.copy()

print(f"Forecasting {n_steps} steps ahead...")

for step in range(n_steps):
    # Reshape for prediction
    input_seq = current_sequence.reshape(1, seq_length, 1)

    # Predict next step
    next_pred = keras_model.predict(input_seq, verbose=0)[0, 0]

    # Store prediction
    predictions.append(next_pred)

    # Update sequence (shift and add new prediction)
    current_sequence = np.roll(current_sequence, -1)
    current_sequence[-1] = next_pred

# Inverse transform
predictions = np.array(predictions).reshape(-1, 1)
predictions_inv = scaler.inverse_transform(predictions).flatten()

print(f"\nMulti-step predictions:")
for i, pred in enumerate(predictions_inv):
    print(f"  Step {i+1}: {pred:.2f}")

# -------------------------------------------------------------------------
# 5.2 Recursive Multi-Step
# -------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5.2 Recursive Multi-Step Forecasting")
print("-" * 50)

print("(Already implemented above - uses same model recursively)")

# ============================================================================
# 6. BIDIRECTIONAL LSTM
# ============================================================================

print("\n" + "=" * 70)
print("6. BIDIRECTIONAL LSTM")
print("=" * 70)

try:
    from tensorflow.keras.layers import Bidirectional

    print("\n" + "-" * 50)
    print("6.1 Defining Bidirectional LSTM")
    print("-" * 50)

    bi_model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(seq_length, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    bi_model.compile(optimizer='adam', loss='mse')

    print("Bidirectional LSTM:")
    bi_model.summary()

    # Train
    print("\nTraining Bidirectional LSTM...")
    bi_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate
    bi_pred = bi_model.predict(X_test, verbose=0)
    bi_pred_inv = scaler.inverse_transform(bi_pred).flatten()
    bi_test_rmse = np.sqrt(mean_squared_error(y_test_inv, bi_pred_inv))

    print(f"\nBidirectional LSTM Test RMSE: {bi_test_rmse:.4f}")

except ImportError:
    print("TensorFlow required")
    bi_test_rmse = None

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("7. MODEL COMPARISON")
print("=" * 70)

print("\n" + "-" * 50)
print("Deep Learning Models Performance")
print("-" * 50)

models_results = []

if lstm_test_rmse is not None:
    models_results.append(('PyTorch LSTM', lstm_test_rmse))

if keras_test_rmse is not None:
    models_results.append(('Keras LSTM', keras_test_rmse))

if gru_test_rmse is not None:
    models_results.append(('GRU', gru_test_rmse))

if bi_test_rmse is not None:
    models_results.append(('Bidirectional LSTM', bi_test_rmse))

# Sort by RMSE
models_results.sort(key=lambda x: x[1])

print("\nModel Rankings:")
for rank, (name, rmse) in enumerate(models_results, 1):
    print(f"  {rank}. {name}: RMSE = {rmse:.4f}")

# ============================================================================
# 8. COMPLETE PIPELINE
# ============================================================================

print("\n" + "=" * 70)
print("8. COMPLETE TIME SERIES FORECASTING PIPELINE")
print("=" * 70)

def time_series_pipeline(data, seq_length=20, epochs=30):
    """
    Complete pipeline for time series forecasting with LSTM.

    Parameters:
    -----------
    data : array-like
        Time series data
    seq_length : int
        Lookback window
    epochs : int
        Training epochs

    Returns:
    --------
    model, scaler, history
    """
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    # Create sequences
    X, y = create_sequences(scaled.flatten(), seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    # Evaluate
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)

    train_rmse = np.sqrt(mean_squared_error(
        scaler.inverse_transform(y_train.reshape(-1, 1)),
        scaler.inverse_transform(train_pred)
    ))
    test_rmse = np.sqrt(mean_squared_error(
        scaler.inverse_transform(y_test.reshape(-1, 1)),
        scaler.inverse_transform(test_pred)
    ))

    print(f"\nPipeline Results:")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")

    return model, scaler, history

# Run pipeline
print("Running complete pipeline...")
model, scaler, history = time_series_pipeline(train_data, seq_length=20, epochs=30)

# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Data Preparation
   - Scale data (0-1 or standardization)
   - Create sequences (sliding window)
   - Train/test split (maintain temporal order)

2. LSTM Architecture
   - Input: (samples, timesteps, features)
   - return_sequences: For stacking LSTMs
   - Dropout: Prevent overfitting

3. Training Tips
   - Use early stopping
   - Monitor validation loss
   - Start with simple models

4. Multi-step Forecasting
   - Direct: Predict all steps at once
   - Iterative: Predict one step, use for next

5. Model Variants
   - LSTM: Long-term dependencies
   - GRU: Simpler, faster
   - Bidirectional: Uses context from both directions

6. Comparison
   - Deep learning often outperforms traditional methods
   - But requires more data and tuning
   - ARIMA still competitive for many tasks

Next: Prophet and Modern Tools (if available)
""")

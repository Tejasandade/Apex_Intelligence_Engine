"""
Apex Intelligence Engine V5 — Deep Learning LSTM Model
=======================================================
Replaces point-in-time XGBoost predictions with sequential order-flow understanding.
Designed specifically for AMD GPUs via Microsoft DirectML.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Attempt to load AMD DirectML
try:
    import torch_directml
    HAS_DML = True
except ImportError:
    HAS_DML = False

from src.core.logging import get_logger

logger = get_logger("apex.models.lstm")


class ApexLSTMNetwork(nn.Module):
    """PyTorch LSTM Architecture for 1-minute Crypto data."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected head
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ApexLSTMModel:
    """Wrapper to integrate PyTorch LSTM with the Apex Engine."""
    
    def __init__(self, name: str, feature_columns: list[str], seq_len: int = 60):
        self.model_name = name
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.input_size = len(feature_columns)
        
        # AMD GPU Setup (4GB VRAM constraint managed via batch size later)
        if HAS_DML:
            self.device = torch_directml.device()
            logger.info("lstm_device_selected", device="DirectML (AMD Radeon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("lstm_device_selected", device="CUDA (NVIDIA)")
        else:
            self.device = torch.device("cpu")
            logger.info("lstm_device_selected", device="CPU")
            
        self.network = ApexLSTMNetwork(input_size=self.input_size).to(self.device)
        self.metadata = {"positive_rate": 0.5, "accuracy": 0.0}
        
        # Internal state buffer for live predictions (must hold last N candles)
        self._history_buffer = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
        """Train the LSTM. Note: Data must be converted to sequences first."""
        logger.info("lstm_training_started", samples=len(X_train), device=str(self.device))
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Convert DataFrames to sequential PyTorch Tensors
        # Due to 8GB system RAM / 4GB VRAM, we must use small batch sizes and careful slicing
        X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values)
        X_val_seq, y_val_seq = self._create_sequences(X_val.values, y_val.values)
        
        train_data = TensorDataset(X_train_seq, y_train_seq)
        val_data = TensorDataset(X_val_seq, y_val_seq)
        
        # 4GB VRAM constraint: Batch size MAX 64 for sequence data
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        
        epochs = 5
        best_val_loss = float("inf")
        
        for epoch in range(epochs):
            self.network.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.network(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            # Validation Phase
            self.network.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.network(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    predictions = (outputs >= 0.5).float()
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
                    
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total if total > 0 else 0
            
            logger.info(f"epoch_{epoch+1}", train_loss=avg_train_loss, val_loss=avg_val_loss, val_accuracy=val_accuracy)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
        self.metadata["positive_rate"] = float(y_train.mean())
        self.metadata["accuracy"] = val_accuracy
        self.metadata["log_loss"] = best_val_loss
        
        return {"accuracy": val_accuracy, "log_loss": best_val_loss}

    def _create_sequences(self, X_data: np.ndarray, y_data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert 2D tabular data into 3D sequence data for LSTM."""
        xs, ys = [], []
        # Memory-efficient sliding window
        for i in range(len(X_data) - self.seq_len):
            xs.append(X_data[i : i + self.seq_len])
            ys.append(y_data[i + self.seq_len])
            
        return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)

    def predict(self, feature_row: pd.DataFrame) -> float:
        """
        Live prediction hook. Maintains a sliding window of the last `seq_len` candles.
        If the window isn't full, returns the baseline probability.
        """
        row_values = feature_row[self.feature_columns].values[0]
        self._history_buffer.append(row_values)
        
        # Keep buffer trimmed to seq_len
        if len(self._history_buffer) > self.seq_len:
            self._history_buffer.pop(0)
            
        # Not enough data for a full sequence yet
        if len(self._history_buffer) < self.seq_len:
            return self.metadata.get("positive_rate", 0.50)
            
        # Prepare tensor
        seq_array = np.array(self._history_buffer, dtype=np.float32)
        seq_tensor = torch.tensor(seq_array).unsqueeze(0).to(self.device)  # Add batch dimension
        
        self.network.eval()
        with torch.no_grad():
            output = self.network(seq_tensor)
            probability = output.item()
            
        return probability

    def save(self, path: Path | str) -> None:
        """Save PyTorch weights and metadata."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"{self.model_name}.pth"
        meta_path = save_dir / f"{self.model_name}_meta.json"
        
        torch.save(self.network.state_dict(), model_path)
        
        with open(meta_path, "w") as f:
            json.dump({
                "features": self.feature_columns,
                "metadata": self.metadata,
                "seq_len": self.seq_len
            }, f, indent=2)

    @classmethod
    def load(cls, name: str, path: Path | str) -> ApexLSTMModel | None:
        """Load PyTorch weights and metadata."""
        save_dir = Path(path)
        model_path = save_dir / f"{name}.pth"
        meta_path = save_dir / f"{name}_meta.json"
        
        if not model_path.exists() or not meta_path.exists():
            return None
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        instance = cls(name=name, feature_columns=meta["features"], seq_len=meta.get("seq_len", 60))
        instance.metadata = meta["metadata"]
        
        instance.network.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.network.eval()
        
        return instance

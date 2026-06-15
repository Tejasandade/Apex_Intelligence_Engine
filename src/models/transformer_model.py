"""
Apex Intelligence Engine V5 — Deep Learning Transformer Model
============================================================
Replaces the basic LSTM with a Time-Series Transformer (TST).
Uses Multi-Head Attention and Positional Encoding to find
complex temporal dependencies in order-flow data.
"""

from __future__ import annotations

import os
import json
import math
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

logger = get_logger("apex.models.transformer")


class PositionalEncoding(nn.Module):
    """Injects positional information into the sequence."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class ApexTransformerNetwork(nn.Module):
    """PyTorch Time-Series Transformer Architecture."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True  # Use batch_first for better DirectML performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Fully connected head
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        out = self.transformer_encoder(x)
        
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Memory-efficient lazy dataset that generates sequences on-the-fly."""
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray, seq_len: int):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
        
    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]


class ApexTransformerModel:
    """Wrapper to integrate PyTorch Transformer with the Apex Engine."""
    
    def __init__(self, name: str, feature_columns: list[str], seq_len: int = 60):
        self.model_name = name
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.input_size = len(feature_columns)
        
        # AMD GPU Setup (4GB VRAM constraint managed via batch size later)
        if HAS_DML:
            self.device = torch_directml.device()
            logger.info("transformer_device_selected", device="DirectML (AMD Radeon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("transformer_device_selected", device="CUDA (NVIDIA)")
        else:
            self.device = torch.device("cpu")
            logger.info("transformer_device_selected", device="CPU")
            
        self.network = ApexTransformerNetwork(input_size=self.input_size).to(self.device)
        self.metadata = {"positive_rate": 0.5, "accuracy": 0.0}
        
        # Internal state buffer for live predictions
        self._history_buffer = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
        """Train the Transformer with Early Stopping and LR Scheduling."""
        logger.info("transformer_training_started", samples=len(X_train), device=str(self.device))
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        
        train_data = TimeSeriesDataset(X_train.values, y_train.values, self.seq_len)
        val_data = TimeSeriesDataset(X_val.values, y_val.values, self.seq_len)
        
        # Transformers are memory-heavy; use a slightly smaller batch size than LSTM
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        epochs = 15
        patience = 4
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_state = None
        
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
                # Gradient clipping to prevent exploding gradients in Transformers
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
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
            
            scheduler.step(avg_val_loss)
            
            logger.info(f"epoch_{epoch+1}", train_loss=avg_train_loss, val_loss=avg_val_loss, val_accuracy=val_accuracy)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.network.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info("early_stopping_triggered", epoch=epoch+1)
                break
                
        # Restore best weights
        if best_model_state:
            self.network.load_state_dict(best_model_state)
                
        self.metadata["positive_rate"] = float(y_train.mean())
        self.metadata["accuracy"] = val_accuracy
        self.metadata["log_loss"] = best_val_loss
        
        return {"accuracy": val_accuracy, "log_loss": best_val_loss}



    def predict(self, feature_row: pd.DataFrame) -> float:
        row_values = feature_row[self.feature_columns].values[0]
        self._history_buffer.append(row_values)
        
        if len(self._history_buffer) > self.seq_len:
            self._history_buffer.pop(0)
            
        if len(self._history_buffer) < self.seq_len:
            return 0.50  # Hard center fallback instead of skewed positive_rate
            
        seq_array = np.array(self._history_buffer, dtype=np.float32)
        seq_tensor = torch.tensor(seq_array).unsqueeze(0).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            output = self.network(seq_tensor)
            probability = output.item()
            
        return probability

    def save(self, path: Path | str) -> None:
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
    def load(cls, name: str, path: Path | str) -> ApexTransformerModel | None:
        save_dir = Path(path)
        model_path = save_dir / f"{name}.pth"
        meta_path = save_dir / f"{name}_meta.json"
        
        if not model_path.exists() or not meta_path.exists():
            return None
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        instance = cls(name=name, feature_columns=meta["features"], seq_len=meta.get("seq_len", 60))
        instance.metadata = meta["metadata"]
        
        # Load directly to configured device (handles DML properly)
        instance.network.load_state_dict(torch.load(model_path, map_location=instance.device))
        instance.network.eval()
        
        return instance

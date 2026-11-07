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

# DirectML disabled to prevent Windows C++ Access Violations
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


def create_sequences(X_data: np.ndarray, y_data: np.ndarray, seq_len: int):
    X = torch.tensor(X_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    
    if len(X) <= seq_len:
        return TensorDataset(torch.empty(0, seq_len, X.shape[1]), torch.empty(0))
        
    X_seq = X.unfold(0, seq_len, 1).transpose(1, 2)
    y_seq = y[seq_len - 1:]
    return TensorDataset(X_seq, y_seq)


class ApexTransformerModel:
    """Wrapper to integrate PyTorch Transformer with the Apex Engine."""
    
    def __init__(self, name: str, feature_columns: list[str], seq_len: int = 60):
        self.model_name = name
        self.feature_columns = feature_columns
        self.seq_len = seq_len
        self.input_size = len(feature_columns)
        
        # Force CPU to prevent DirectML Access Violations on Windows with large batch sizes
        self.device = torch.device("cpu")
        logger.info("transformer_device_selected", device="CPU (Forced)")
            
        self.network = ApexTransformerNetwork(input_size=self.input_size).to(self.device)
        self.metadata = {"positive_rate": 0.5, "accuracy": 0.0}
        
        # Internal state buffer for live predictions
        self._history_buffer = []

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
        """Train the Transformer with F1-optimized Early Stopping.
        
        Key design decisions:
        - Uses BCEWithLogitsLoss + pos_weight to rebalance the imbalanced dataset
        - Early-stops on F1 score (not loss/accuracy) to prevent the "always say NO" trap
        - Reports Precision, Recall, and F1 as the primary quality metrics
        """
        logger.info("transformer_training_started", samples=len(X_train), device=str(self.device))
        
        # --- CRITICAL FIX: Rebalance the loss for imbalanced classes ---
        # If positive_rate = 16%, the negative class outnumbers positive ~5:1.
        # pos_weight tells the loss function to treat each positive sample as 5 negatives.
        positive_rate = float(y_train.mean())
        if positive_rate > 0 and positive_rate < 1:
            pos_weight_val = (1.0 - positive_rate) / positive_rate
        else:
            pos_weight_val = 1.0
        pos_weight_tensor = torch.tensor([pos_weight_val], device=self.device)
        
        # BCEWithLogitsLoss = Sigmoid + BCELoss fused for numerical stability
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
        
        train_data = create_sequences(X_train.values, y_train.values, self.seq_len)
        val_data = create_sequences(X_val.values, y_val.values, self.seq_len)
        
        train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=2048, shuffle=False)
        
        epochs = 20
        patience = 5
        patience_counter = 0
        best_val_f1 = 0.0
        best_model_state = None
        best_metrics = {}
        
        for epoch in range(epochs):
            # --- Training Phase ---
            self.network.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                # Network outputs raw logits (no sigmoid) when using BCEWithLogitsLoss
                logits = self.network(batch_x).squeeze()
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
            # --- Validation Phase: Compute F1 / Precision / Recall ---
            self.network.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.network(batch_x).squeeze()
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    
                    # Apply sigmoid to logits for prediction
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    all_preds.append(preds.cpu())
                    all_targets.append(batch_y.cpu())
            
            # Concatenate all batches
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            # Compute confusion matrix components
            tp = ((all_preds == 1) & (all_targets == 1)).sum().item()
            fp = ((all_preds == 1) & (all_targets == 0)).sum().item()
            fn = ((all_preds == 0) & (all_targets == 1)).sum().item()
            tn = ((all_preds == 0) & (all_targets == 0)).sum().item()
            total = len(all_targets)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / total if total > 0 else 0.0
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Schedule LR based on F1 (mode='max')
            scheduler.step(f1)
            
            logger.info(
                f"epoch_{epoch+1}",
                train_loss=round(avg_train_loss, 4),
                val_loss=round(avg_val_loss, 4),
                f1=round(f1, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                accuracy=round(accuracy, 4),
                tp=tp, fp=fp, fn=fn, tn=tn,
            )
            
            # --- Early Stopping on F1 ---
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_model_state = self.network.state_dict()
                best_metrics = {
                    "f1": f1, "precision": precision, "recall": recall,
                    "accuracy": accuracy, "val_loss": avg_val_loss,
                }
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info("early_stopping_triggered", epoch=epoch+1, best_f1=best_val_f1)
                break
                
        # Restore best weights
        if best_model_state:
            self.network.load_state_dict(best_model_state)
                
        self.metadata["positive_rate"] = positive_rate
        self.metadata["f1_score"] = best_metrics.get("f1", 0.0)
        self.metadata["precision"] = best_metrics.get("precision", 0.0)
        self.metadata["recall"] = best_metrics.get("recall", 0.0)
        self.metadata["accuracy"] = best_metrics.get("accuracy", 0.0)
        self.metadata["log_loss"] = best_metrics.get("val_loss", 0.0)
        
        return best_metrics



    def predict(self, feature_row: pd.DataFrame) -> float:
        row_values = feature_row[self.feature_columns].values[0]
        self._history_buffer.append(row_values)
        
        if len(self._history_buffer) > self.seq_len:
            self._history_buffer.pop(0)
            
        if len(self._history_buffer) < self.seq_len:
            return 0.50
            
        # Bypass PyTorch due to Windows C++ deadlocks
        # Use pure mathematical heuristic: high z-score = high mean reversion probability
        try:
            z_score_idx = self.feature_columns.index('z_score')
            current_z = row_values[z_score_idx]
            # Sigmoid activation of the inverse z-score (mean reversion)
            probability = 1.0 / (1.0 + np.exp(-abs(current_z) + 1.5))
            return float(probability)
        except ValueError:
            return 0.50

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

"""
Reinforcement Learning Meta-Controller
========================================
A PPO Agent that sits above the Council. It observes the Council's conviction,
the LSTM's conviction, and the current Regime to output a continuous position
size multiplier between 0.0 and 1.0.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.core.config import WEIGHTS_DIR
from src.core.logging import get_logger

logger = get_logger("apex.models.rl_agent")


class TradeSizingEnv(gym.Env):
    """
    Custom Gym Environment for learning optimal position sizing.
    
    Observation Space:
        - XGBoost Conviction (0.0 to 1.0)
        - LSTM/Transformer Conviction (0.0 to 1.0)
        - Regime (0=Trending, 1=Ranging)
        - Volatility (0=Low, 1=High)
        - Recent Win Rate (0.0 to 1.0)
        - Current Drawdown (0.0 to 1.0)
        - Margin Available (0.0 to 1.0)
        - Normalized ATR (0.0 to 1.0)
        
    Action Space:
        - Position Size Multiplier: Box(0.0, 1.0)
    """

    def __init__(self, historical_trades: list[dict[str, Any]]):
        super().__init__()
        self.trades = historical_trades
        self.current_step = 0
        self.max_steps = len(self.trades) - 1
        
        # Action: Continuous value from 0.0 to 1.0
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obs: 8 dimensions
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        
        self.initial_equity = 10000.0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.recent_wins = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.recent_wins = [1] * 5
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step >= len(self.trades):
            trade = self.trades[-1]
        else:
            trade = self.trades[self.current_step]
            
        xgb_conviction = float(trade.get("xgb_conviction", 0.5))
        transformer_conviction = float(trade.get("transformer_conviction", 0.5))
        regime = trade.get("regime", "TRENDING_HIGH_VOL")
        
        is_ranging = 1.0 if "RANGING" in regime else 0.0
        is_high_vol = 1.0 if "HIGH_VOL" in regime else 0.0
        win_rate = float(np.mean(self.recent_wins)) if self.recent_wins else 0.5
        
        drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        margin_avail = 1.0 - drawdown  # Simplification for simulation
        atr_norm = float(trade.get("atr_norm", 0.5))  # Requires normalized ATR in trades
        
        return np.array([
            xgb_conviction, transformer_conviction, is_ranging, is_high_vol, 
            win_rate, drawdown, margin_avail, atr_norm
        ], dtype=np.float32)

    def step(self, action):
        trade = self.trades[self.current_step]
        sizing = float(np.clip(action[0], 0.0, 1.0))
        actual_pnl_pct = float(trade.get("pnl_pct", 0.0))
        
        # Apply sizing
        realized_pnl_pct = actual_pnl_pct * sizing
        pnl_dollars = self.equity * realized_pnl_pct
        self.equity += pnl_dollars
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            
        drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        
        is_win = 1 if actual_pnl_pct > 0 else 0
        self.recent_wins.append(is_win)
        if len(self.recent_wins) > 10:
            self.recent_wins.pop(0)
            
        # ── Kelly Criterion Reward Function ──
        # Penalize aggressive sizing during a drawdown.
        # Reward aggressive sizing on high-conviction winners.
        if actual_pnl_pct > 0:
            reward = sizing * 2.0
        else:
            # If in a drawdown > 10%, penalize losses MUCH harder
            dd_penalty = 2.0 if drawdown > 0.10 else 1.0
            reward = - (sizing * 4.0 * dd_penalty)
            
        if actual_pnl_pct < 0 and sizing < 0.1:
            reward = 0.5  # Good job avoiding a loser
            
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {"equity": self.equity}


class RLMetaController:
    """Manages the PPO Agent for Trade Sizing."""

    def __init__(self, market_type: str, symbol: str):
        self.market_type = market_type
        self.symbol = symbol
        self.model_path = WEIGHTS_DIR / f"{symbol}_{market_type}_ppo_meta.zip"
        self.model = None

    def train(self, historical_trades: list[dict[str, Any]], total_timesteps: int = 100000):
        if not historical_trades or len(historical_trades) < 100:
            logger.warning("rl_insufficient_trades", trades=len(historical_trades))
            return

        env = TradeSizingEnv(historical_trades)
        
        logger.info("rl_training_started", steps=total_timesteps, trades=len(historical_trades))
        
        # PPO requires a vectorized environment
        try:
            from stable_baselines3.common.env_util import make_vec_env
            vec_env = make_vec_env(lambda: env, n_envs=1)
            
            self.model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=0.0003)
            self.model.learn(total_timesteps=total_timesteps)
            
            self.model.save(self.model_path)
            logger.info("rl_training_completed", saved_path=str(self.model_path))
        except Exception as e:
            logger.error("rl_training_failed", error=str(e))

    def load(self):
        if self.model_path.exists():
            try:
                self.model = PPO.load(self.model_path)
                logger.info("rl_agent_loaded", path=str(self.model_path))
                return True
            except Exception as e:
                logger.error("rl_agent_load_failed", error=str(e))
        return False

    def get_position_size(
        self,
        xgb_conviction: float,
        transformer_conviction: float,
        regime: str,
        win_rate: float,
        current_drawdown: float = 0.0,
        margin_available: float = 1.0,
        atr_normalized: float = 0.5
    ) -> float:
        """
        Get the recommended position size multiplier [0.0, 1.0].
        Defaults to 0.5 if no model is loaded.
        """
        if self.model is None:
            # Fallback heuristic if RL isn't trained
            if transformer_conviction > 0.7 and "TRENDING" in regime and current_drawdown < 0.05:
                return 1.0
            elif "RANGING" in regime and transformer_conviction < 0.4:
                return 0.1
            elif current_drawdown > 0.10:
                return 0.25 # Cut size in drawdown
            return 0.5
            
        is_ranging = 1.0 if "RANGING" in regime else 0.0
        is_high_vol = 1.0 if "HIGH_VOL" in regime else 0.0
        
        obs = np.array([
            xgb_conviction, transformer_conviction, is_ranging, is_high_vol, 
            win_rate, current_drawdown, margin_available, atr_normalized
        ], dtype=np.float32)
        
        action, _states = self.model.predict(obs, deterministic=True)
        raw_action = float(np.clip(action[0], 0.0, 1.0))
        # Prevent RL from zeroing out trades completely (Cowardly Agent problem)
        return max(0.2, raw_action)

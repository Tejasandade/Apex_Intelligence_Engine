"""
Apex Intelligence Engine V5 — Live Signal Engine (Phase 3)
=============================================================
The real-time brain with full intelligence pipeline:

Phase 1: WebSocket -> Features -> Model -> Signal
Phase 2: + Ensemble + Monitor + Walk-Forward + Regime
Phase 3: + Council (multi-advisor confirmation)

Full pipeline:
  WebSocket -> Features -> Regime -> Ensemble -> Council -> Monitor -> Signal
                                                   |
                                             5 Advisors:
                                             Momentum, Structure,
                                             Volume, Regime, Risk
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import get_model_params
from src.core.logging import get_logger
from src.features.live_store import LiveFeatureStore
from src.council.council import Council
from src.engine.mtf_buffer import MultiTimeframeBuffer
from src.models.ensemble import ModelEnsemble, WeightingMethod
from src.models.monitor import PerformanceMonitor
from src.models.regime import RegimeDetector, MarketRegime
from src.models.walk_forward import WalkForwardRetrainer

logger = get_logger("apex.engine.signal")


@dataclass
class TradingSignal:
    """A trading signal emitted by the signal engine."""

    signal_id: str = ""  # Unique ID for tracking
    symbol: str = ""
    direction: str = ""  # "BUY", "SELL", or "HOLD"
    conviction: float = 0.0  # Model probability [0, 1]
    ensemble_conviction: float = 0.0  # Agreement-based conviction modifier
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    quantity: float = 0.0
    regime: str = ""  # Current market regime
    atr: float = 0.0
    timestamp: float = 0.0  # Unix timestamp (ms)
    features: dict[str, float] = field(default_factory=dict)  # Top features
    member_predictions: dict[str, float] = field(default_factory=dict)
    council_approved: bool = True
    council_score: float = 1.0
    council_explanation: str = ""
    council_votes: dict[str, dict] = field(default_factory=dict)
    council_summary: str = ""
    reason: str = ""


class LiveSignalEngine:
    """
    Real-time signal generation engine with Phase 2 intelligence.

    Pipeline:
    1. LiveFeatureStore → compute features
    2. RegimeDetector → classify market state
    3. ModelEnsemble → weighted voting across models
    4. PerformanceMonitor → track + decay detection
    5. WalkForwardRetrainer → accumulate data for periodic retraining
    6. Signal Decision → thresholding + sizing
    """

    def __init__(
        self,
        symbol: str,
        market_type: str = "crypto",
        models: dict[str, Any] | None = None,
        conviction_threshold: float = 0.60,
        risk_per_trade: float = 200.0,
        default_rr: float = 2.0,
        max_concurrent: int = 1,
        enable_ensemble: bool = True,
        enable_monitoring: bool = True,
        enable_walk_forward: bool = True,
        enable_council: bool = True,
        retrain_interval: int = 1000,
        consensus_threshold: float = 0.60,
    ):
        self.symbol = symbol
        self.market_type = market_type
        self.models = models or {}
        self.conviction_threshold = conviction_threshold
        self.risk_per_trade = risk_per_trade
        self.default_rr = default_rr
        self.max_concurrent = max_concurrent

        # ── Phase 1 Components ───────────────────────────────────────────────
        self.live_store = LiveFeatureStore(market_type=market_type, buffer_size=200)

        # ── Phase 2 Components ───────────────────────────────────────────────
        # Regime detector (stateful, with ATR history)
        self.regime_detector = RegimeDetector()
        
        # HTF Trend Alignment (1H EMAs simulated on 1m data)
        # 1H 20 EMA = 1200 periods on 1m
        # 1H 50 EMA = 3000 periods on 1m
        self._htf_ema_20: float | None = None
        self._htf_ema_50: float | None = None

        # Model ensemble
        self.ensemble: ModelEnsemble | None = None
        self._enable_ensemble = enable_ensemble
        if enable_ensemble and models:
            self.ensemble = ModelEnsemble(
                weighting=WeightingMethod.REGIME,
                disagreement_penalty=0.3,
            )
            for regime_name, model in models.items():
                self.ensemble.add_member(model, regime=regime_name)

        # Performance monitor
        self.monitor: PerformanceMonitor | None = None
        self._enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.monitor = PerformanceMonitor(model_name=f"{symbol}_ensemble")

        # Walk-forward retrainer
        self.retrainer: WalkForwardRetrainer | None = None
        self._enable_walk_forward = enable_walk_forward
        if enable_walk_forward:
            self.retrainer = WalkForwardRetrainer(
                symbol=symbol,
                market_type=market_type,
                retrain_interval=retrain_interval,
            )

        # ── Phase 3 Components ───────────────────────────────────────────────
        # The Council — multi-advisor signal confirmation
        self.council: Council | None = None
        self._enable_council = enable_council
        if enable_council:
            self.council = Council(
                consensus_threshold=consensus_threshold,
                min_voting_advisors=3,
                enable_veto=True,
                scalp_mode=False,  # Strict mode — no loose filters
            )
            self.council.setup_default_advisors()
            
        # ── Phase 6 Components: RL Meta-Controller ───────────────────────────
        self.rl_meta_controller = None
        self._load_rl_meta_controller()

    def _load_rl_meta_controller(self) -> None:
        """Attempt to load the trained RL Meta-Controller."""
        try:
            from stable_baselines3 import PPO
            from src.core.config import DATA_DIR
            model_path = DATA_DIR / "models" / "rl_meta_controller.zip"
            if model_path.exists():
                self.rl_meta_controller = PPO.load(str(model_path))
                if self.ensemble:
                    self.ensemble.weighting = WeightingMethod.RL
                logger.info("rl_meta_controller_loaded", path=str(model_path))
            else:
                logger.debug("rl_meta_controller_not_found")
        except ImportError:
            logger.debug("stable_baselines3_not_installed_skipping_rl")
        except Exception as e:
            logger.warning("failed_to_load_rl_meta_controller", error=str(e))

        # Multi-Timeframe Buffer
        self.mtf_buffer = MultiTimeframeBuffer()

        # ── State ────────────────────────────────────────────────────────────
        self._open_position_count = 0
        self._signals_generated = 0
        self._candles_processed = 0
        self._last_signal: TradingSignal | None = None
        self._pending_signals: dict[str, TradingSignal] = {}
        self._halted = False  # True if critical decay detected
        self._last_adx: float = 0.0  # Track for dashboard
        
        # Deduplication
        self._last_signal_direction = ""
        self._suppression_counter = 0

    @property
    def is_ready(self) -> bool:
        return self.live_store.is_warmed_up and len(self.models) > 0

    @property
    def is_halted(self) -> bool:
        return self._halted

    def warm_up(self, historical_data: pd.DataFrame) -> None:
        """Prime the feature store and retrainer with historical data."""
        self.live_store.warm_up(historical_data)

        # Also seed the walk-forward retrainer
        if self.retrainer:
            self.retrainer.load_historical(historical_data)
            self.retrainer.set_current_models(self.models)

        logger.info(
            "signal_engine_warmed_up",
            symbol=self.symbol,
            buffer=self.live_store.buffer_length,
            models=list(self.models.keys()),
            ensemble=self._enable_ensemble,
            monitor=self._enable_monitoring,
            walk_forward=self._enable_walk_forward,
            council=self._enable_council,
        )

    def set_open_positions(self, count: int) -> None:
        self._open_position_count = count

    async def process_candle(self, candle: dict[str, Any], oracle_levels: list | None = None) -> TradingSignal | None:
        """
        Process a closed candle through the full Phase 2 pipeline.

        Returns:
            TradingSignal if conviction threshold is met, None otherwise.
        """
        self.mtf_buffer.add_candle(candle)
        self._candles_processed += 1
        if self._suppression_counter > 0:
            self._suppression_counter -= 1

        # ── Step 1: Compute features ─────────────────────────────────────────
        features = self.live_store.update(candle)
        if features is None:
            return None

        close = float(candle.get("close", 0))
        if close <= 0:
            return None

        # ── Step 1.5: Update HTF EMAs (O(1) Recursive) ───────────────────────
        if self._htf_ema_20 is None:
            self._htf_ema_20 = close
            self._htf_ema_50 = close
        else:
            alpha_20 = 2.0 / (1200.0 + 1.0)
            alpha_50 = 2.0 / (3000.0 + 1.0)
            self._htf_ema_20 = (close - self._htf_ema_20) * alpha_20 + self._htf_ema_20
            self._htf_ema_50 = (close - self._htf_ema_50) * alpha_50 + self._htf_ema_50

        # ── Step 2: Extract key indicators ───────────────────────────────────
        atr = float(features.iloc[0].get("ATR", 0.0))
        adx = float(features.iloc[0].get("ADX", 0.0))
        chop = float(features.iloc[0].get("CHOP", 50.0))
        self._last_adx = adx  # Track for dashboard

        if atr <= 0:
            return None

        # ── Step 3: Classify regime (stateful) ───────────────────────────────
        regime = self.regime_detector.update(adx=adx, chop=chop, atr=atr)

        # Track regime in monitor
        if self.monitor:
            self.monitor.record_regime(regime.value)

        # ── Step 4: Feed walk-forward retrainer ──────────────────────────────
        if self.retrainer:
            should_retrain = self.retrainer.add_candle(candle)
            if should_retrain:
                logger.info("walk_forward_triggered", candles=self._candles_processed)
                retrain_result = await self.retrainer.retrain()
                if retrain_result.get("status") == "success":
                    # Hot-swap models
                    new_models = self.retrainer.get_current_models()
                    if new_models:
                        self.models.update(new_models)
                        self._rebuild_ensemble()
                        logger.info(
                            "models_hot_swapped",
                            updated=retrain_result.get("models_updated", []),
                        )

                    # Clear decay alerts after retraining
                    if self.monitor:
                        self.monitor.clear_alerts()
                        self._halted = False

        # ── Step 5: Check if halted due to decay ─────────────────────────────
        if self._halted:
            logger.debug("signal_suppressed_halted", candle=self._candles_processed)
            return None

        # ── Step 6: Predict ──────────────────────────────────────────────────
        probability: float
        ensemble_conviction: float = 1.0
        member_predictions: dict[str, float] = {}
        
        # ── RL Meta-Controller DISABLED ──
        # RL was trained on random model outputs — adjusts noise, not signal.
        # Reintroduce only after base model proves profitable.
        rl_risk_multiplier = 1.0

        if self.ensemble and self._enable_ensemble:
            # Use ensemble prediction
            probability, ensemble_conviction, member_predictions = (
                self.ensemble.predict(features, current_regime=regime.value)
            )
            baselines = [
                m.model.metadata.get("metrics", {}).get("positive_rate", 0.5)
                for m in self.ensemble._members
            ]
            baseline = sum(baselines) / len(baselines) if baselines else 0.50
        else:
            # Fallback to single model
            model = self._select_model(regime)
            if model is None:
                return None
            try:
                probability = model.predict(features)
                baseline = model.metadata.get("metrics", {}).get("positive_rate", 0.5)
            except Exception as e:
                logger.error("signal_predict_error", error=str(e))
                return None

        # ── Step 7: Signal decision (Raw Probability) ─────────────────────────
        # Use raw model probability directly. No re-centering, no baseline adjustment.
        # With asymmetric labeling (TP=1.5x, SL=2.0x), baseline ~52.6%.
        # Model just needs to be slightly better than random.
        buy_threshold = self.conviction_threshold  # e.g., 0.55
        sell_threshold = 1.0 - self.conviction_threshold  # e.g., 0.45
        
        macro_downtrend = False
        macro_uptrend = False

        if self._htf_ema_20 is not None and self._htf_ema_50 is not None:
            if self._htf_ema_20 < self._htf_ema_50 and close < self._htf_ema_20:
                macro_downtrend = True
            elif self._htf_ema_20 > self._htf_ema_50 and close > self._htf_ema_20:
                macro_uptrend = True

        direction = "HOLD"
        if probability >= buy_threshold:
            direction = "BUY"
        elif probability <= sell_threshold:
            direction = "SELL"
            
        # ── Strict Macro Trend Suppression with Ultimate Quantitative Override ──
        if direction == "BUY" and macro_downtrend:
            if isinstance(features, pd.DataFrame):
                sweep_signal = float(features["liquidity_sweep_signal"].iloc[-1]) if "liquidity_sweep_signal" in features else 0.0
                sweep_1h = float(features["Liquidity_Sweep_1H"].iloc[-1]) if "Liquidity_Sweep_1H" in features else 0.0
                ob_dist = float(features["OB_bull_dist"].iloc[-1]) if "OB_bull_dist" in features else 100.0
                vwap_z = float(features["VWAP_zscore"].iloc[-1]) if "VWAP_zscore" in features else 0.0
                ofi = float(features["order_flow_imbalance"].iloc[-1]) if "order_flow_imbalance" in features else 0.0
            else:
                sweep_signal = float(features.get("liquidity_sweep_signal", 0.0))
                sweep_1h = float(features.get("Liquidity_Sweep_1H", 0.0))
                ob_dist = float(features.get("OB_bull_dist", 100.0))
                vwap_z = float(features.get("VWAP_zscore", 0.0))
                ofi = float(features.get("order_flow_imbalance", 0.0))
                
            is_structural_sweep = (sweep_signal > 0.5) or (sweep_1h > 0.5) or (ob_dist < 0.002)
            is_statistically_extended = vwap_z < -2.0
            is_volumetric_absorption = ofi > 0.5
            
            if is_structural_sweep and is_statistically_extended and is_volumetric_absorption:
                logger.warning("macro_override_sweep", reason="ULTIMATE OVERRIDE: Sweep + VWAP Z-Score Extension + Volume Absorption detected. Catching the knife.")
            else:
                logger.info("signal_suppressed", reason="Macro Downtrend (Falling Knife Filter)", price=close)
                direction = "HOLD"

        elif direction == "SELL" and macro_uptrend:
            if isinstance(features, pd.DataFrame):
                sweep_signal = float(features["liquidity_sweep_signal"].iloc[-1]) if "liquidity_sweep_signal" in features else 0.0
                sweep_1h = float(features["Liquidity_Sweep_1H"].iloc[-1]) if "Liquidity_Sweep_1H" in features else 0.0
                ob_dist = float(features["OB_bear_dist"].iloc[-1]) if "OB_bear_dist" in features else 100.0
                vwap_z = float(features["VWAP_zscore"].iloc[-1]) if "VWAP_zscore" in features else 0.0
                ofi = float(features["order_flow_imbalance"].iloc[-1]) if "order_flow_imbalance" in features else 0.0
            else:
                sweep_signal = float(features.get("liquidity_sweep_signal", 0.0))
                sweep_1h = float(features.get("Liquidity_Sweep_1H", 0.0))
                ob_dist = float(features.get("OB_bear_dist", 100.0))
                vwap_z = float(features.get("VWAP_zscore", 0.0))
                ofi = float(features.get("order_flow_imbalance", 0.0))
                
            is_structural_sweep = (sweep_signal < -0.5) or (sweep_1h < -0.5) or (ob_dist < 0.002)
            is_statistically_extended = vwap_z > 2.0
            is_volumetric_absorption = ofi < -0.5
            
            if is_structural_sweep and is_statistically_extended and is_volumetric_absorption:
                logger.warning("macro_override_sweep", reason="ULTIMATE OVERRIDE: Sweep + VWAP Z-Score Extension + Volume Absorption detected. Catching the knife.")
            else:
                logger.info("signal_suppressed", reason="Macro Uptrend", price=close)
                direction = "HOLD"

        if direction == "HOLD":
            return None
            
        # ── Step 7.5: Signal Deduplication ───────────────────────────────────
        if direction == self._last_signal_direction and self._suppression_counter > 0:
            return None

        # ── Step 8: Apply ensemble conviction modifier ───────────────────────
        effective_conviction = probability * ensemble_conviction
        # If models strongly disagree, skip the trade
        if ensemble_conviction < 0.5 and self._enable_ensemble:
            logger.debug(
                "signal_suppressed_disagreement",
                conviction=f"{ensemble_conviction:.4f}",
                predictions=member_predictions,
            )
            return None

        # ── Step 8.2: Minimum effective conviction filter (REMOVED) ──────────
        # Removed hardcoded 0.65 filter. We want signals that pass the 0.60 
        # conviction_threshold to be evaluated by the Council and shown on the dashboard.

        # ── Step 8.5: Council evaluation (Phase 3) ───────────────────────────
        council_approved = True
        council_score = 1.0
        council_explanation = ""
        trade_type = "SWING"  # Default if no council
        council_votes = {}
        council_summary = ""

        # Determine effective regime using MTF macro override
        effective_regime = regime.value
        if self.mtf_buffer.is_ready():
            df_15m = self.mtf_buffer.get_15m_df()
            if df_15m is not None and len(df_15m) >= 14:
                # Calculate basic ADX manually for 15m to see macro trend
                from src.features.indicators.technical import compute_adx
                try:
                    df_15m_adx = compute_adx(df_15m, period=14)
                    adx_15m = df_15m_adx.iloc[-1]
                    if adx_15m > 30.0 and effective_regime in ("RANGING", "QUIET"):
                        logger.debug("macro_override_to_swing", micro_regime=effective_regime, adx_15m=adx_15m)
                        effective_regime = "TRENDING"
                except Exception:
                    pass

        if self.council and self._enable_council:
            # ── SNIPER OVERRIDE ──────────────────────────────────────────────
            if isinstance(features, pd.DataFrame):
                ofi = float(features["order_flow_imbalance"].iloc[-1]) if "order_flow_imbalance" in features else 0.0
            else:
                ofi = float(features.get("order_flow_imbalance", 0.0))
                
            is_sniper_buy = direction == "BUY" and ofi > 0.8
            is_sniper_sell = direction == "SELL" and ofi < -0.8
            
            if is_sniper_buy or is_sniper_sell:
                logger.warning("sniper_override_activated", direction=direction, ofi=ofi)
                council_approved = True
                council_score = 1.0
                council_explanation = "SNIPER OVERRIDE: Extreme Order Flow Imbalance"
                trade_type = "SCALP"
                council_summary = "Council Bypassed by Sniper Override"
                council_votes = {"Sniper": {"vote": "APPROVE", "conviction": 1.0}}
            else:
                council_decision = self.council.evaluate(
                    features=features,
                    direction=direction,
                    price=close,
                    regime=effective_regime,
                    atr=atr,
                    df_5m=self.mtf_buffer.get_5m_df() if self.mtf_buffer.is_ready() else None,
                    df_15m=self.mtf_buffer.get_15m_df() if self.mtf_buffer.is_ready() else None,
                    mtf_ready=self.mtf_buffer.is_ready(),
                    oracle_levels=oracle_levels or [],
                )
                council_approved = council_decision.approved
                council_score = council_decision.consensus_score
                council_explanation = council_decision.explanation
                trade_type = council_decision.trade_type
                council_summary = council_decision.vote_summary
                council_votes = {
                    v.advisor_name: {"vote": v.vote.value, "conviction": v.conviction} 
                    for v in council_decision.votes
                }

                if not council_approved:
                    logger.info(
                        "signal_blocked_by_council",
                        direction=direction,
                        consensus=f"{council_score:.4f}",
                        summary=council_decision.vote_summary,
                    )
                    # We do NOT return None here anymore. We return the blocked signal 
                    # so the dashboard can see it. LiveRunner will handle the block.

        # ── Step 9: Check capacity ───────────────────────────────────────────
        if self._open_position_count >= self.max_concurrent:
            return None

        # ── Step 10: Calculate levels (trade_type and conviction-aware) ──────────
        # Dynamic Risk/Reward based on conviction (from Implementation Plan)
        if council_score >= 0.85:
            # Home Run Setup: Huge runners
            rr_ratio = 4.0
            stop_mult = 2.0 if trade_type == "SWING" else 1.5
        elif council_score >= 0.70:
            # Strong Setup: Extended runners
            rr_ratio = 3.0
            stop_mult = 2.0 if trade_type == "SWING" else 1.5
        else:
            # Standard Setup
            rr_ratio = 2.5
            stop_mult = 2.0 if trade_type == "SWING" else 1.5
        
        # Strict ATR-based stop loss (No minimum percentage override)
        # This keeps losses strictly proportional to volatility, allowing larger position sizes 
        # and bigger profits when ATR is tight.
        if direction == "BUY":
            stop_loss = close - (atr * stop_mult)
            take_profit = 0.0  # NO HARD TP! Hold for structural trailing stop
            stop_distance = close - stop_loss
        else:
            stop_loss = close + (atr * stop_mult)
            take_profit = 0.0  # NO HARD TP! Hold for structural trailing stop
            stop_distance = stop_loss - close
        
        # ── Conviction-scaled position sizing (Multi-Tier Kelly Style) ───────
        base_risk = self.risk_per_trade
        if council_score >= 0.85:
            # Home Run Setup
            council_risk_multiplier = 1.5
        elif council_score >= 0.70:
            # Strong Setup
            council_risk_multiplier = 1.2
        else:
            # Standard Setup
            council_risk_multiplier = 1.0
            
        # Blend Council Risk with RL Meta-Controller Risk
        final_risk_multiplier = council_risk_multiplier * rl_risk_multiplier
            
        risk_this_trade = base_risk * final_risk_multiplier
        quantity = risk_this_trade / stop_distance if stop_distance > 0 else 0
        if quantity <= 0:
            return None

        # ── Step 11: Build signal ────────────────────────────────────────────
        signal_id = f"sig_{self.symbol}_{uuid.uuid4().hex[:8]}"

        # Top features
        top_features = {}
        if self.ensemble:
            importance = self.ensemble.get_feature_importance()
        else:
            model = self._select_model(regime)
            importance = model.get_feature_importance() if model else {}

        if importance:
            sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for fname, _ in sorted_feats:
                top_features[fname] = float(features.iloc[0].get(fname, 0.0))

        signal = TradingSignal(
            signal_id=signal_id,
            symbol=self.symbol,
            direction=direction,
            conviction=probability,
            ensemble_conviction=ensemble_conviction,
            price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            regime=effective_regime,
            atr=atr,
            timestamp=candle.get("timestamp", time.time() * 1000),
            features=top_features,
            member_predictions=member_predictions,
            council_approved=council_approved,
            council_score=council_score,
            council_explanation=council_explanation,
            council_votes=council_votes,
            council_summary=council_summary,
            reason=(
                f"{regime.value} regime | p={probability:.4f} "
                f"| agreement={ensemble_conviction:.2f} "
                f"| council={council_score:.2f} | ATR={atr:.2f}"
            ),
        )

        # ── Step 12: Record in monitor ───────────────────────────────────────
        if self.monitor:
            self.monitor.record_prediction(
                prediction_id=signal_id,
                probability=probability,
                direction=direction,
            )
            self._pending_signals[signal_id] = signal

        # ── Step 13: Finalize ────────────────────────────────────────────────
        self._signals_generated += 1
        self._last_signal = signal
        self._last_signal_direction = direction
        
        # Suppress duplicate signals
        if signal.council_approved:
            self._suppression_counter = 5
        else:
            self._suppression_counter = 2

        logger.info(
            "signal_generated",
            signal_id=signal.signal_id,
            symbol=self.symbol,
            direction=direction,
            prob=f"{probability:.4f}",
            regime=regime.value,
            council=council_approved,
            approved=signal.council_approved,
        )

        return signal

    def resolve_signal(self, signal_id: str, profitable: bool, pnl: float = 0.0) -> None:
        """
        Resolve a signal after the trade closes.
        Updates the performance monitor with the outcome.
        """
        if self.monitor:
            alerts = self.monitor.resolve_prediction(signal_id, profitable, pnl)

            # Check for critical decay
            if self.monitor.has_critical_alerts:
                logger.warning(
                    "model_decay_critical",
                    model=self.monitor.model_name,
                    accuracy=f"{self.monitor.current_accuracy:.4f}",
                )
                self._halted = True

        self._pending_signals.pop(signal_id, None)

    def _select_model(self, regime: MarketRegime) -> Any | None:
        """Select the appropriate model for the current regime."""
        regime_name = regime.value.lower()
        if regime_name in self.models:
            return self.models[regime_name]

        if regime in (MarketRegime.VOLATILE, MarketRegime.TRENDING):
            if "trending" in self.models:
                return self.models["trending"]
        elif regime in (MarketRegime.QUIET, MarketRegime.RANGING):
            if "ranging" in self.models:
                return self.models["ranging"]

        if self.models:
            return next(iter(self.models.values()))
        return None

    def _rebuild_ensemble(self) -> None:
        """Rebuild the ensemble after model hot-swap."""
        if not self._enable_ensemble:
            return

        self.ensemble = ModelEnsemble(
            weighting=WeightingMethod.REGIME,
            disagreement_penalty=0.3,
        )
        for regime_name, model in self.models.items():
            self.ensemble.add_member(model, regime=regime_name)

        logger.info("ensemble_rebuilt", members=self.ensemble.num_members)

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive engine statistics."""
        stats = {
            "symbol": self.symbol,
            "ready": self.is_ready,
            "halted": self._halted,
            "candles_processed": self._candles_processed,
            "signals_generated": self._signals_generated,
            "models_loaded": list(self.models.keys()),
            "buffer_length": self.live_store.buffer_length,
            "warmed_up": self.live_store.is_warmed_up,
            "current_regime": self.regime_detector.current_regime.value,
            "pending_signals": len(self._pending_signals),
        }

        if self.ensemble:
            stats["ensemble_members"] = self.ensemble.num_members

        if self.monitor:
            stats["monitor"] = self.monitor.get_stats()

        if self.retrainer:
            stats["retrainer"] = self.retrainer.get_stats()

        if self._last_signal:
            stats["last_signal"] = {
                "direction": self._last_signal.direction,
                "conviction": self._last_signal.conviction,
                "price": self._last_signal.price,
                "regime": self._last_signal.regime,
            }

        return stats

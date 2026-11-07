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
from src.models.statistical_alpha import StatisticalAlphaModel

logger = get_logger("apex.engine.signal")


@dataclass
class TradingSignal:
    """A trading signal emitted by the signal engine."""

    signal_id: str = ""  # Unique ID for tracking
    symbol: str = ""
    direction: str = ""  # "BUY", "SELL", or "HOLD"
    strategy_type: str = "UNKNOWN" # "BREAKOUT" or "MEAN_REVERSION"
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
        conviction_threshold: float = 0.58,
        risk_per_trade: float = 200.0,
        default_rr: float = 2.0,
        max_concurrent: int = 1,
        enable_ensemble: bool = True,
        enable_monitoring: bool = True,
        enable_walk_forward: bool = True,
        enable_council: bool = True,
        enable_statistical_alpha: bool = True,
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

        self.ensemble_long: ModelEnsemble | None = None
        self.ensemble_short: ModelEnsemble | None = None
        self._enable_ensemble = enable_ensemble
        
        if enable_ensemble and models:
            self.ensemble_long = ModelEnsemble(weighting=WeightingMethod.REGIME, disagreement_penalty=0.3)
            self.ensemble_short = ModelEnsemble(weighting=WeightingMethod.REGIME, disagreement_penalty=0.3)
            
            for regime_name, model in models.items():
                r_lower = regime_name.lower()
                if "_long" in r_lower:
                    self.ensemble_long.add_member(model, regime=r_lower.replace("_long", ""))
                elif "_short" in r_lower:
                    self.ensemble_short.add_member(model, regime=r_lower.replace("_short", ""))
                else:
                    self.ensemble_long.add_member(model, regime=r_lower)
                    self.ensemble_short.add_member(model, regime=r_lower)

        # Statistical Alpha Engine (Phase 7)
        self.statistical_model: StatisticalAlphaModel | None = None
        self._enable_statistical_alpha = enable_statistical_alpha
        if enable_statistical_alpha:
            self.statistical_model = StatisticalAlphaModel(name=f"{symbol}_statistical_alpha")

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
            )
            if self._enable_statistical_alpha:
                self.council.setup_statistical_advisors()
            elif market_type == "crypto":
                self.council.setup_default_advisors(symbol=symbol)
            else:
                self.council.setup_india_advisors()
            
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
                if getattr(self, "ensemble_long", None):
                    self.ensemble_long.weighting = WeightingMethod.RL
                if getattr(self, "ensemble_short", None):
                    self.ensemble_short.weighting = WeightingMethod.RL
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

    def warm_up(
        self,
        historical_data: pd.DataFrame,
        df_15m: pd.DataFrame | None = None,
        df_1h: pd.DataFrame | None = None,
        df_4h: pd.DataFrame | None = None,
    ) -> None:
        """Prime the feature store, MTF buffer, and retrainer with historical data."""
        self.live_store.warm_up(historical_data)
        self.mtf_buffer.seed_direct(df_5m=historical_data, df_15m=df_15m, df_1h=df_1h, df_4h=df_4h)

        # Warm up HTF EMAs using the historical data (O(N) initial pass)
        if not historical_data.empty and "close" in historical_data.columns:
            closes = historical_data["close"].values
            # Initialize with first value
            self._htf_ema_20 = closes[0]
            self._htf_ema_50 = closes[0]
            
            alpha_20 = 2.0 / (1200.0 + 1.0)
            alpha_50 = 2.0 / (3000.0 + 1.0)
            
            # Apply EMA recursively to build up the true value
            for c in closes[1:]:
                self._htf_ema_20 = (c - self._htf_ema_20) * alpha_20 + self._htf_ema_20
                self._htf_ema_50 = (c - self._htf_ema_50) * alpha_50 + self._htf_ema_50

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
        
        # Feed the statistical model raw candles
        if self.statistical_model:
            self.statistical_model.add_candle(candle)
            
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
                        self._rebuild_ensembles()
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
        probability: float = 0.5
        ensemble_conviction: float = 1.0
        member_predictions: dict[str, float] = {}
        
        rl_risk_multiplier = 1.0
        
        prob_long = 0.5
        prob_short = 0.5
        conv_long = 1.0
        conv_short = 1.0

        if self._enable_statistical_alpha and self.statistical_model:
            # Override ML predictions entirely with Pure Mathematics
            prob_long, prob_short, strat_type = self.statistical_model.predict()
            member_predictions["strategy_type"] = strat_type
            conv_long = 1.0
            conv_short = 1.0
            ensemble_conviction = 1.0
        elif self._enable_ensemble:
            if self.ensemble_long and self.ensemble_long._members:
                prob_long, conv_long, _ = self.ensemble_long.predict(features, current_regime=regime.value)
            if self.ensemble_short and self.ensemble_short._members:
                prob_short, conv_short, _ = self.ensemble_short.predict(features, current_regime=regime.value)
        else:
            # Fallback (Legacy)
            model = self._select_model(regime)
            if model is not None:
                try:
                    prob = model.predict(features)
                    prob_long = prob
                    prob_short = 1.0 - prob
                except Exception as e:
                    logger.error("signal_predict_error", error=str(e))
                    return None

        # ── Step 7: Signal decision (True Dual-Probability) ─────────────────
        buy_threshold = self.conviction_threshold
        sell_threshold = self.conviction_threshold
        
        macro_downtrend = False
        macro_uptrend = False

        if self._htf_ema_20 is not None and self._htf_ema_50 is not None:
            if self._htf_ema_20 < self._htf_ema_50 and close < self._htf_ema_20:
                macro_downtrend = True
            elif self._htf_ema_20 > self._htf_ema_50 and close > self._htf_ema_20:
                macro_uptrend = True

        direction = "HOLD"
        
        if prob_long >= buy_threshold and prob_long > prob_short:
            direction = "BUY"
            probability = prob_long
            ensemble_conviction = conv_long
        elif prob_short >= sell_threshold and prob_short > prob_long:
            direction = "SELL"
            probability = prob_short
            ensemble_conviction = conv_short
            
        # ── Phase 6: Strict Macro Trend Filter ──
        # Guarantee 0 flaws by never fighting the HTF Trend
        if direction == "BUY" and macro_downtrend:
            direction = "HOLD"
            logger.info("signal_suppressed", reason="Strict Macro Downtrend Block", price=close)

        elif direction == "SELL" and macro_uptrend:
            direction = "HOLD"
            logger.info("signal_suppressed", reason="Strict Macro Uptrend Block", price=close)

        if direction == "HOLD":
            return None
            
        # ── Step 7.5: Signal Deduplication ───────────────────────────────────
        if direction == self._last_signal_direction and self._suppression_counter > 0:
            return None

        # ── Step 8: Apply ensemble conviction modifier ───────────────────────
        effective_conviction = probability * ensemble_conviction

        # ── REMOVED: effective_conviction gate was killing 100% of signals ──
        # Math: prob=0.62 * ensemble=0.87 = 0.539 < 0.55 = KILLED
        # The conviction_threshold (0.58) already filters weak signals.
        # if effective_conviction < 0.55:
        #     return None

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
            # All signals go through the Council's democratic process. No overrides.
            strat_type = member_predictions.get("strategy_type", "UNKNOWN")
            council_decision = self.council.evaluate(
                features=features,
                direction=direction,
                price=close,
                regime=effective_regime,
                atr=atr,
                df_5m=self.mtf_buffer.get_5m_df() if self.mtf_buffer.is_ready() else None,
                df_15m=self.mtf_buffer.get_15m_df() if self.mtf_buffer.is_ready() else None,
                df_1h=self.mtf_buffer.get_1h_df() if self.mtf_buffer.is_ready() else None,
                df_4h=self.mtf_buffer.get_4h_df() if self.mtf_buffer.is_ready() else None,
                mtf_ready=self.mtf_buffer.is_ready(),
                oracle_levels=oracle_levels or [],
                strategy_type=strat_type,
            )
            council_approved = council_decision.approved
            council_score = council_decision.consensus_score
            council_explanation = council_decision.explanation
            trade_type = council_decision.trade_type
            council_summary = council_decision.vote_summary
            council_votes = {
                v.advisor_name: {
                    "vote": v.vote.value, 
                    "conviction": v.conviction,
                    "weight": v.weight
                } 
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
        
        # Strict ATR-based stop loss with HARD MINIMUM TAKE PROFIT
        # Every trade MUST have a positive R:R. The trailing stop can extend
        # winners beyond the hard TP, but we guarantee a minimum profit target.
        if direction == "BUY":
            stop_loss = close - (atr * stop_mult)
            stop_distance = close - stop_loss
            take_profit = close + (stop_distance * rr_ratio)  # Hard minimum TP
        else:
            stop_loss = close + (atr * stop_mult)
            stop_distance = stop_loss - close
            take_profit = close - (stop_distance * rr_ratio)  # Hard minimum TP
        
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
        target_ensemble = self.ensemble_long if direction == "BUY" else self.ensemble_short
        if target_ensemble and hasattr(target_ensemble, "get_feature_importance"):
            importance = target_ensemble.get_feature_importance()
        else:
            model = self._select_model(regime)
            importance = getattr(model, "get_feature_importance", lambda: {})() if model else {}

        if importance:
            sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for fname, _ in sorted_feats:
                top_features[fname] = float(features.iloc[0].get(fname, 0.0))

        signal = TradingSignal(
            signal_id=signal_id,
            symbol=self.symbol,
            direction=direction,
            strategy_type=member_predictions.get("strategy_type", "UNKNOWN"),
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
        
        # Suppress duplicate signals — reduced from 5/2 to 3/1 (P2 improvement)
        # 3 candles = 15min cooldown for approved; 1 candle = 5min for blocked
        if signal.council_approved:
            self._suppression_counter = 3
        else:
            self._suppression_counter = 1

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

    def _select_model(self, regime: Any) -> Any | None:
        """Select the appropriate model for the current regime."""
        regime_str = regime.value if hasattr(regime, 'value') else str(regime)
        if "TRENDING" in regime_str.upper():
            return self.models.get("TRENDING")
        elif "RANGING" in regime_str.upper() or "QUIET" in regime_str.upper():
            return self.models.get("RANGING")

        if self.models:
            return next(iter(self.models.values()))
        return None

    def _rebuild_ensembles(self) -> None:
        """Rebuild the ensembles after a model update."""
        if not self._enable_ensemble or not self.models:
            return

        self.ensemble_long = ModelEnsemble(weighting=WeightingMethod.REGIME, disagreement_penalty=0.3)
        self.ensemble_short = ModelEnsemble(weighting=WeightingMethod.REGIME, disagreement_penalty=0.3)
        
        for regime_name, model in self.models.items():
            r_lower = regime_name.lower()
            if "_long" in r_lower:
                self.ensemble_long.add_member(model, regime=r_lower.replace("_long", ""))
            elif "_short" in r_lower:
                self.ensemble_short.add_member(model, regime=r_lower.replace("_short", ""))
            else:
                self.ensemble_long.add_member(model, regime=r_lower)
                self.ensemble_short.add_member(model, regime=r_lower)

        logger.info("ensemble_rebuilt", members=max(self.ensemble_long.num_members, self.ensemble_short.num_members))

    def hot_swap_models(self, new_models: dict) -> None:
        """
        Thread-safely replace ensemble members with freshly retrained models.
        Called by RetrainScheduler.check_pending_swap() on the main thread.
        """
        import threading
        if not hasattr(self, "_swap_lock"):
            self._swap_lock = threading.Lock()

        with self._swap_lock:
            try:
                # Merge new models into self.models dict
                for name, model in new_models.items():
                    self.models[name] = model

                # Rebuild ensembles from updated model dict
                self._rebuild_ensembles()
                logger.info(
                    "hot_swap_complete",
                    new_models=list(new_models.keys()),
                    total_models=len(self.models),
                )
            except Exception as exc:
                logger.warning(f"hot_swap_models failed: {exc}")

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

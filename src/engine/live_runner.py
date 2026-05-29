"""
Apex Intelligence Engine V5 — Live Runner (Phase 4)
======================================================
Main orchestrator for live trading/paper trading.

Full pipeline:
    WebSocket → Features → Regime → Ensemble → Council → Position Manager → Broker

Phase 4 additions:
- Paper Broker: Full simulation with slippage + commission
- Position Manager: Stop loss/take profit monitoring on every tick
- Risk feedback loop: Position Manager → Council Risk Advisor
- Trade recording: Full trade history with P&L tracking
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import DATA_DIR, get_model_params
from src.core.logging import get_logger
from src.data.providers.binance import BinanceHistoricalProvider
from src.data.providers.binance_ws import BinanceWebSocket
from src.engine.signal_engine import LiveSignalEngine, TradingSignal
from src.execution.paper_broker import PaperBroker
from src.execution.position_manager import PositionManager
from src.execution.broker import PositionSide
from src.dashboard.server import DashboardServer
from src.models.xgboost_model import ApexXGBoostModel
from src.features.store import FeatureStore

logger = get_logger("apex.engine.live_runner")

MODELS_DIR = DATA_DIR / "models"


class LiveRunner:
    """
    Main live trading orchestrator (Phase 4).

    Full lifecycle:
    1. Load models → create signal engines
    2. Initialize broker + position manager
    3. Connect WebSocket
    4. On each candle: predict → council → execute → manage
    5. On each tick: update positions, check stops
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        market_type: str = "crypto",
        paper_mode: bool = True,
        initial_balance: float = 10_000.0,
        max_positions: int = 3,
        max_drawdown_pct: float = 10.0,
        daily_loss_limit: float = 500.0,
        dashboard_port: int = 8765,
        enable_dashboard: bool = True,
    ):
        self.symbols = [s.lower() for s in (symbols or ["btcusdt"])]
        self.market_type = market_type
        self.paper_mode = paper_mode

        # Components (initialized in setup)
        self._ws: BinanceWebSocket | None = None
        self._signal_engines: dict[str, LiveSignalEngine] = {}
        self._running = False

        # Phase 4: Execution components
        self._broker: PaperBroker | None = None
        self._position_mgr: PositionManager | None = None
        self._initial_balance = initial_balance
        self._max_positions = max_positions
        self._max_drawdown_pct = max_drawdown_pct
        self._daily_loss_limit = daily_loss_limit

        # Phase 5: Dashboard server
        self._dashboard: DashboardServer | None = None
        self._enable_dashboard = enable_dashboard
        self._dashboard_port = dashboard_port

        # Signal log
        self._signal_log: list[dict[str, Any]] = []
        self._start_time: float = 0.0

    async def setup(self) -> None:
        """Load models, init broker, warm up feature stores, start dashboard."""
        logger.info("live_runner_setup", symbols=self.symbols, market_type=self.market_type)
        print(f"\n[DIAGNOSTICS] Starting Live Runner setup for {self.symbols} in {self.market_type} mode...")

        # ── Phase 5: Start dashboard server ───────────────────────────────────
        if self._enable_dashboard:
            self._dashboard = DashboardServer(port=self._dashboard_port)
            await self._dashboard.start()

        # ── Phase 4: Initialize broker + position manager ─────────────────────
        if self.paper_mode:
            self._broker = PaperBroker(
                initial_balance=self._initial_balance,
                commission_bps=10.0,
                slippage_bps=5.0,
            )
            await self._broker.connect()

        if self._broker:
            self._position_mgr = PositionManager(
                broker=self._broker,
                max_positions=self._max_positions,
                max_drawdown_pct=self._max_drawdown_pct,
                daily_loss_limit=self._daily_loss_limit,
            )
            await self._position_mgr.initialize()

        # ── Load signal engines ───────────────────────────────────────────────
        for symbol in self.symbols:
            models = self._load_models(symbol)
            if not models:
                logger.error("no_models_found", symbol=symbol)
                print(f"[ERROR] No models found for {symbol}. Engine will not start for this symbol.")
                continue

            print(f"[DIAGNOSTICS] Successfully loaded {len(models)} models for {symbol}: {list(models.keys())}")

            engine = LiveSignalEngine(
                symbol=symbol,
                market_type=self.market_type,
                models=models,
                conviction_threshold=0.60,
                risk_per_trade=200.0,
                default_rr=2.0,
            )

            await self._warm_up_engine(engine, symbol)
            self._signal_engines[symbol] = engine
            logger.info(
                "signal_engine_ready",
                symbol=symbol,
                models=list(models.keys()),
                buffer=engine.live_store.buffer_length,
            )

        if not self._signal_engines:
            print("[ERROR] No signal engines could be initialized. Aborting setup.")
            raise RuntimeError("No signal engines could be initialized.")

        # ── WebSocket ─────────────────────────────────────────────────────────
        print("[DIAGNOSTICS] Connecting to WebSocket stream...")
        self._ws = BinanceWebSocket(symbols=self.symbols)
        self._ws.on_candle = self._on_candle
        self._ws.on_tick = self._on_tick  # Phase 4: monitor stops on every tick

        logger.info(
            "live_runner_ready",
            engines=list(self._signal_engines.keys()),
            paper_mode=self.paper_mode,
            broker=self._broker.broker_name if self._broker else "none",
            balance=f"${self._initial_balance:,.2f}",
        )
        print("[DIAGNOSTICS] Setup complete. Ready to receive real-time data.")

    def _load_models(self, symbol: str) -> dict[str, Any]:
        """Load trained models for a symbol from disk."""
        models: dict[str, Any] = {}
        feature_store = FeatureStore(self.market_type)

        for regime in ["trending", "ranging"]:
            model_name = f"{symbol}_{self.market_type}_{regime}"
            model_path = MODELS_DIR / f"{model_name}.json"

            if not model_path.exists():
                continue

            model = ApexXGBoostModel(
                name=model_name,
                feature_columns=feature_store.feature_columns,
            )
            model.load(MODELS_DIR)
            models[regime] = model
            logger.info("model_loaded", name=model_name, trained=model.is_trained)

        return models

    async def _warm_up_engine(self, engine: LiveSignalEngine, symbol: str) -> None:
        """Fetch recent history and warm up the feature store."""
        cache_path = DATA_DIR / "historical" / "crypto" / f"{symbol}.parquet"

        if cache_path.exists():
            logger.info("warming_from_cache", path=str(cache_path))
            df = pd.read_parquet(cache_path)
            engine.warm_up(df.tail(200))
            return

        logger.info("warming_from_api", symbol=symbol)
        provider = BinanceHistoricalProvider()
        try:
            await provider.connect()
            df = await provider.fetch_historical_candles(
                symbol=symbol.upper(), interval="1m", limit=200,
            )
            if not df.empty:
                engine.warm_up(df)
        finally:
            await provider.disconnect()

    async def _on_tick(self, symbol: str, candle: dict[str, Any]) -> None:
        """
        Called on EVERY WebSocket tick (not just closed candles).
        Used for real-time stop loss / take profit monitoring.
        """
        if self._position_mgr is None:
            return

        price = float(candle.get("close", 0))
        if price <= 0:
            return

        # Update position manager with latest price
        main_trade, partial_trade = self._position_mgr.update_price(symbol, price)

        for trade in (partial_trade, main_trade):
            if trade is not None:
                # A stop loss, take profit, or scale-out was hit!
                self._handle_trade_close(trade)

                # Feed result back to signal engine monitor (only for full closes or if we want to count partials)
                # Usually we just resolve signal on the main trade close
                if trade == main_trade:
                    engine = self._signal_engines.get(symbol)
                    if engine:
                        engine.resolve_signal(
                            trade.signal_id,
                            profitable=trade.pnl > 0,
                            pnl=trade.pnl,
                        )
        if self._dashboard:
            pos = self._position_mgr._positions.get(symbol)
            unrealized_pnl = pos.unrealized_pnl if pos and pos.quantity > 0 else 0.0
            self._dashboard.broadcast_tick(unrealized_pnl, price)

        # Feed risk state back to Council
        self._update_council_risk()

    async def _on_candle(self, symbol: str, candle: dict[str, Any]) -> None:
        """Callback for each closed candle from WebSocket."""
        engine = self._signal_engines.get(symbol)
        if engine is None:
            return

        # Update position count in signal engine
        if self._position_mgr:
            engine.set_open_positions(self._position_mgr.open_position_count)

        signal = await engine.process_candle(candle)

        # Update trailing stops on candle close
        if self._position_mgr:
            close_price = float(candle.get("close", 0))
            trail_update = self._position_mgr.update_on_candle_close(
                symbol=symbol,
                close_price=close_price,
                atr=engine._last_signal.atr if engine._last_signal else 0.0, # Approximate
                regime=engine.regime_detector.current_regime.value
            )
            if trail_update and self._dashboard:
                self._dashboard.broadcast_trail_update(trail_update)

        # Always broadcast status to dashboard on every candle
        if self._dashboard:
            self._broadcast_dashboard_status()

        if signal is not None:
            await self._handle_signal(signal)

    async def _handle_signal(self, signal: TradingSignal) -> None:
        """Handle a generated trading signal — execute via position manager."""
        mode = "PAPER" if self.paper_mode else "LIVE"

        # Log signal
        signal_dict = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": signal.symbol,
            "direction": signal.direction,
            "conviction": signal.conviction,
            "ensemble_agreement": signal.ensemble_conviction,
            "council_score": signal.council_score,
            "price": signal.price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "quantity": signal.quantity,
            "regime": signal.regime,
            "atr": signal.atr,
        }
        self._signal_log.append(signal_dict)
        self._save_signal_log()

        # ── Dashboard: Broadcast Council Votes ───────────────────────────────
        if self._dashboard:
            if signal.council_votes:
                self._dashboard.broadcast_council(
                    votes=signal.council_votes,
                    consensus=signal.council_score,
                    approved=signal.council_approved,
                    summary=signal.council_summary,
                )
            
            # Broadcast ensemble predictions
            trending_prob = 0.5
            ranging_prob = 0.5
            for k, v in signal.member_predictions.items():
                if "trending" in k: trending_prob = v
                if "ranging" in k: ranging_prob = v

            self._dashboard.broadcast_ensemble(
                trending=float(trending_prob),
                ranging=float(ranging_prob),
                agreement=signal.ensemble_conviction,
            )

        # ── Handle Blocked Signals ───────────────────────────────────────────
        if not signal.council_approved:
            if self._dashboard:
                self._dashboard.broadcast_signal(
                    direction=signal.direction, symbol=signal.symbol,
                    price=signal.price, conviction=signal.conviction,
                    council_score=signal.council_score, reason=signal.reason,
                    blocked=True,
                )
            return

        # ── Phase 4: Execute via position manager ─────────────────────────────
        if self._position_mgr and self._broker:
            self._broker.update_price(signal.symbol, signal.price)

            order = await self._position_mgr.open_position(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=signal.quantity,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                regime=signal.regime,
                council_score=signal.council_score,
                atr=signal.atr,
            )

            if order and order.status.value == "FILLED":
                arrow = "▲" if signal.direction == "BUY" else "▼"
                print(
                    f"\n  [{mode}] {arrow} {signal.direction} {signal.symbol.upper()} "
                    f"@ ${order.filled_price:,.2f}"
                )
                print(f"    Conviction: {signal.conviction:.4f} | Regime: {signal.regime}")
                print(f"    Stop: ${signal.stop_loss:,.2f} | Target: ${signal.take_profit:,.2f}")
                print(f"    Qty: {order.filled_quantity:.6f} | ATR: ${signal.atr:.2f}")
                print(f"    Council: {signal.council_score:.2f} | Commission: ${order.commission:.4f}")
                print(f"    Reason: {signal.reason}")
                print()

                # Broadcast to dashboard
                if self._dashboard:
                    self._dashboard.broadcast_signal(
                        direction=signal.direction, symbol=signal.symbol,
                        price=order.filled_price, conviction=signal.conviction,
                        council_score=signal.council_score, reason=signal.reason,
                    )
                    self._dashboard.broadcast_trade_open(
                        symbol=signal.symbol, direction=signal.direction,
                        entry_price=order.filled_price,
                        stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    )
                    # Broadcast ensemble + council
                    self._dashboard.broadcast_ensemble(
                        trending=signal.member_predictions.get('btcusdt_crypto_trending', 0.5),
                        ranging=signal.member_predictions.get('btcusdt_crypto_ranging', 0.5),
                        agreement=signal.ensemble_conviction,
                    )
            else:
                reason = "risk blocked" if order else "position manager blocked"
                logger.info("signal_not_executed", reason=reason, signal=signal.signal_id)
        else:
            # Fallback: print-only mode
            arrow = "▲" if signal.direction == "BUY" else "▼"
            print(
                f"\n  [{mode}] {arrow} {signal.direction} {signal.symbol.upper()} "
                f"@ ${signal.price:,.2f}"
            )
            print(f"    Conviction: {signal.conviction:.4f} | Regime: {signal.regime}")
            print(f"    Stop: ${signal.stop_loss:,.2f} | Target: ${signal.take_profit:,.2f}")
            print(f"    Qty: {signal.quantity:.6f} | ATR: ${signal.atr:.2f}")
            print(f"    Reason: {signal.reason}")
            print()

    def _handle_trade_close(self, trade) -> None:
        """Handle a trade that was closed (stop/TP hit)."""
        pnl_icon = "💰" if trade.pnl > 0 else "🔻"
        print(
            f"\n  {pnl_icon} TRADE CLOSED: {trade.symbol.upper()} "
            f"({trade.exit_reason.upper()})"
        )
        print(
            f"    Entry: ${trade.entry_price:,.2f} → Exit: ${trade.exit_price:,.2f}"
        )
        print(
            f"    P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.4%}) | "
            f"Duration: {trade.duration_seconds:.0f}s"
        )

        # Print running totals
        if self._position_mgr:
            stats = self._position_mgr.get_stats()
            print(
                f"    Balance: ${self._broker._balance:,.2f} | "
                f"Win rate: {stats['win_rate']:.0%} | "
                f"Trades: {stats['total_trades']}"
            )
        print()

        # Broadcast to dashboard
        if self._dashboard:
            self._dashboard.broadcast_trade_close(
                symbol=trade.symbol, side=trade.side,
                entry_price=trade.entry_price, exit_price=trade.exit_price,
                pnl=trade.pnl, pnl_pct=trade.pnl_pct,
                exit_reason=trade.exit_reason,
                duration_seconds=trade.duration_seconds,
            )

    def _update_council_risk(self) -> None:
        """Feed position manager risk state to Council's Risk advisor."""
        if not self._position_mgr:
            return

        risk_state = self._position_mgr.get_risk_state()

        for engine in self._signal_engines.values():
            if engine.council:
                engine.council.update_risk_state(**risk_state)

    def _save_signal_log(self) -> None:
        """Save signals to CSV for analysis."""
        if not self._signal_log:
            return
        log_dir = DATA_DIR / "live_signals"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"signals_{time.strftime('%Y%m%d')}.csv"
        pd.DataFrame(self._signal_log).to_csv(log_path, index=False)

    async def run(self) -> None:
        """Run the live trading loop (blocks forever)."""
        print("\n[DIAGNOSTICS] Live Runner starting...")
        if not self._signal_engines:
            await self.setup()

        self._running = True
        self._start_time = time.time()

        broker_info = ""
        if self._broker:
            broker_info = f"\n  Balance: ${self._initial_balance:,.2f}"
            broker_info += f"\n  Max DD:  {self._max_drawdown_pct}%"

        print("\n" + "=" * 60)
        print("  APEX INTELLIGENCE ENGINE V5 — LIVE MODE")
        print("=" * 60)
        print(f"  Mode:    {'PAPER' if self.paper_mode else 'LIVE'}")
        print(f"  Symbols: {', '.join(s.upper() for s in self.symbols)}")
        print(f"  Models:  {sum(len(e.models) for e in self._signal_engines.values())} loaded")
        if broker_info:
            print(broker_info)
        print(f"  Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        stats_task = asyncio.create_task(self._print_stats_loop())

        try:
            print("[DIAGNOSTICS] Launching WebSocket connection...")
            await self._ws.connect()
        except KeyboardInterrupt:
            print("\n[DIAGNOSTICS] Received KeyboardInterrupt, shutting down...")
            pass
        except asyncio.CancelledError:
            print("\n[DIAGNOSTICS] Received CancelledError, shutting down...")
            pass
        except Exception as e:
            print(f"\n[ERROR] Fatal error in live runner: {e}")
            logger.error("live_runner_fatal_error", error=str(e))
        finally:
            self._running = False
            stats_task.cancel()
            await self.shutdown()

    def _broadcast_dashboard_status(self) -> None:
        """Send current system status to the dashboard."""
        if not self._dashboard:
            return

        total_signals = sum(e._signals_generated for e in self._signal_engines.values())
        total_candles = sum(e._candles_processed for e in self._signal_engines.values())

        # Collect regime and ADX from first engine
        regime = ''
        adx = 0.0
        for engine in self._signal_engines.values():
            regime = engine.regime_detector.current_regime.value
            adx = engine._last_adx
            break

        balance = self._initial_balance
        total_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        drawdown_pct = 0.0
        cooldown_remaining = 0
        trailing_stops = {}

        if self._position_mgr and self._broker:
            balance = self._broker._balance
            pm_stats = self._position_mgr.get_stats()
            total_pnl = pm_stats.get('total_pnl', 0.0)
            total_trades = pm_stats.get('total_trades', 0)
            winning_trades = pm_stats.get('winning_trades', 0)
            drawdown_pct = pm_stats.get('max_drawdown_pct', 0.0)
            cooldown_remaining = pm_stats.get('cooldown_remaining', 0)
            trailing_stops = pm_stats.get('trailing_stops', {})

        self._dashboard.broadcast_status(
            balance=balance,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            drawdown_pct=drawdown_pct,
            regime=regime,
            adx=adx,
            signals=total_signals,
            candles=total_candles,
            cooldown_remaining=cooldown_remaining,
            trailing_stops=trailing_stops,
        )

    async def _print_stats_loop(self) -> None:
        """Print status every 60 seconds and broadcast to dashboard."""
        while self._running:
            await asyncio.sleep(60)
            if not self._running:
                break

            uptime = time.time() - self._start_time
            total_signals = sum(e._signals_generated for e in self._signal_engines.values())
            total_candles = sum(e._candles_processed for e in self._signal_engines.values())
            ws_stats = self._ws.get_stats() if self._ws else {}

            status = (
                f"  [STATUS] Uptime: {uptime / 60:.1f}min | "
                f"Candles: {total_candles} | "
                f"Signals: {total_signals} | "
                f"WS ticks: {ws_stats.get('ticks_received', 0)} | "
                f"Connected: {ws_stats.get('connected', False)}"
            )

            if self._position_mgr:
                pm_stats = self._position_mgr.get_stats()
                status += (
                    f"\n  [P&L]    Balance: ${self._broker._balance:,.2f} | "
                    f"Trades: {pm_stats['total_trades']} | "
                    f"Win: {pm_stats['win_rate']:.0%} | "
                    f"PnL: ${pm_stats['total_pnl']:+,.2f} | "
                    f"DD: {pm_stats['max_drawdown_pct']:.1f}%"
                )

            print(status)

            # Also broadcast to dashboard
            self._broadcast_dashboard_status()

    async def shutdown(self) -> None:
        """Gracefully shut down all components."""
        logger.info("live_runner_shutting_down")

        if self._dashboard:
            await self._dashboard.stop()

        if self._ws:
            await self._ws.disconnect()

        # Close any open positions
        if self._position_mgr:
            for symbol in self.symbols:
                await self._position_mgr.close_position(symbol, reason="shutdown")

        if self._broker:
            await self._broker.disconnect()

        self._save_signal_log()

        uptime = time.time() - self._start_time if self._start_time > 0 else 0
        total_signals = sum(e._signals_generated for e in self._signal_engines.values())

        print(f"\n  Shutdown complete. Uptime: {uptime / 60:.1f} minutes. "
              f"Signals generated: {total_signals}")

        if self._position_mgr:
            stats = self._position_mgr.get_stats()
            print(f"  Final P&L: ${stats['total_pnl']:+,.2f} | "
                  f"Trades: {stats['total_trades']} | "
                  f"Win rate: {stats['win_rate']:.0%}")

        logger.info(
            "live_runner_stopped",
            uptime_min=f"{uptime / 60:.1f}",
            signals=total_signals,
        )

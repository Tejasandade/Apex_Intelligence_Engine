# Apex Intelligence Engine — V6 Master Roadmap
**A Complete Architectural Response & Implementation Plan**

---

## Preamble: Where We Stand

The Apex V5 engine is already ahead of 95% of retail quant systems. The Council architecture, Isotonic calibration, Triple-Barrier labeling, and regime-splitting are not beginner features — they reflect genuine systems thinking. What follows is not a critique of V5. It is the honest next chapter: what to build, in what order, why, and what each upgrade actually costs you in complexity and risk.

The four proposed upgrades in the current implementation plan are all valid. But the **order** in which they are executed, and the **conditions** under which each is unlocked, will determine whether this system becomes a reliable edge or an over-engineered machine that breaks under live conditions.

This document formalizes the complete roadmap.

---

## Upgrade 1 — The Oracle (Advanced HTF SMC Memory)

### The Problem (Confirmed)
The current engine is temporally blind at boot. It loads 200 one-minute candles and begins trading with no awareness of the macro landscape. Price could be walking into a 4-Hour Order Block or a Daily FVG — levels that institutional players actively defend — and the engine would take the trade with full confidence.

### The Solution (Approved with Modifications)

Build the Oracle as a boot-time, read-only context layer. It does not trade. It does not vote. It **informs** the `StructureAdvisor` with macro context before any vote is cast.

**Boot Sequence:**
1. On engine start, Oracle downloads the last 30 days of 1H and 4H OHLCV data from the broker API.
2. It maps all persistent structural levels: Order Blocks (bullish and bearish), HTF Fair Value Gaps, Daily Open/High/Low, and Weekly Open.
3. It stores these levels in a simple in-memory dictionary keyed by `(level_type, price_zone_low, price_zone_high)`.
4. The `StructureAdvisor` is upgraded with a new check: before casting its vote, it queries the Oracle to determine if current 1m price is within a configurable proximity threshold (e.g., 0.2% of ATR) of any HTF level.
5. If price is **approaching** a counter-directional HTF level, the StructureAdvisor reduces its vote weight. If price is **interacting with** a confluent HTF level (same direction as the trade), it increases its weight.

**Critical Implementation Note:** Order Blocks are inherently definitional — your code's definition of an OB is the only one that matters. Before going live with the Oracle, run a mandatory validation backtest: compare the win rate of trades that had HTF Oracle confluence versus those that did not. If the delta is less than 3%, the Oracle is adding architectural complexity without edge and the threshold logic needs to be revised.

**Oracle Refresh:** The Oracle should re-run every 4 hours during live operation to invalidate mitigated levels (levels that price has already traded through).

---

## Upgrade 2 — Structural Trailing Stop ("Diamond Hands")

### The Problem (Confirmed)
ATR-based trailing is rigid by definition. It is calibrated to current volatility, not market structure. In a strong trend with normal volatility pulses, the ATR stop gets hit on corrections that are structurally irrelevant — the trend continues without the position.

### The Solution (Approved with a Hybrid Guard)

Rewrite `SmartTrailingStop` to use structural swing points as the trailing anchor. However, a pure structural trailing approach has one critical vulnerability: **in a fast impulsive move, swing highs can form very close to price**, and the structural stop can be just as tight — or tighter — than ATR. The solution is a hybrid with a stage gate.

**Implementation Stages:**

**Stage 1 — Entry to +1.0R: ATR Trailing (Capital Protection Mode)**
The original ATR-based trailing remains active until the trade has moved 1.0R in our favor (meaning risk is now covered by profit). During this stage, we are purely protecting capital.

**Stage 2 — +1.0R to +2.0R: Hybrid Mode**
The stop is set to the *wider* of: ATR trail OR the last confirmed swing high/low on the 5m chart. This prevents the structural stop from being tighter than ATR during fast moves.

**Stage 3 — Beyond +2.0R: Full Structural Mode ("Diamond Hands")**
Once the trade is 2R+ in profit, the stop fully transitions to structural trailing. It tucks above the most recent 5m or 15m swing high (for a short) and only moves when a new, lower swing high forms. The position is only exited if:
- Price creates a confirmed CHOCH (Change of Character) against the position on the 5m chart, **or**
- A hard maximum drawdown from the peak of the trade is hit (e.g., give back no more than 40% of open profit from the peak).

**The peak-drawdown rule is non-negotiable.** Without it, structural trailing can hold a position through a full reversal while the engine waits for a CHOCH that may form 10R lower than the high.

---

## Upgrade 3 — Pyramiding (Adding to Winners)

### The Problem (Confirmed)
Single-entry positions leave significant compounding potential on the table during strong directional trends. A system that correctly identifies trend direction but enters once and exits once will always underperform a system that compounds into that same trend.

### The Solution (Approved with Strict Unlock Conditions)

Upgrade `PositionManager` to support pyramiding tranches. However, this is the highest-risk upgrade in this roadmap and must be gated behind strict conditions. **Pyramiding should not be activated until Upgrades 1 and 2 have been running live on PaperBroker for a minimum of 30 trading days with a positive Sharpe ratio.**

**Pyramiding Rules:**

**Unlock Condition:** The existing position's trailing stop must have moved past the entry price. The position must be provably risk-free before a second tranche is opened. This is not negotiable.

**Tranche Sizing:** The second tranche is sized at 50% of the original position size. Not equal. Not larger. Half size. The reason is asymmetric: if the trend continues, the combined position performs excellently. If the trend reverses immediately after pyramiding, the second tranche adds a small loss while the first is still profitable.

**Maximum Tranches:** Hard limit of 2 tranches (original + 1 add). No more. The compounding benefit of a third tranche does not justify the complexity and risk of managing three separate stop levels.

**Tranche Entry Conditions:** The `SignalEngine` must generate a fresh signal (not just price movement) for the second tranche to be entered. Both the ML model (>0.52 probability) and the Council (>62% consensus, higher than the standard 60% threshold) must independently approve the second entry.

**Trailing:** Both tranches share a single structural trailing stop level. The stop is not split. This simplifies management and ensures both positions exit simultaneously on a structural break.

**Risk Advisor Override:** The Risk Advisor's VETO power applies to both tranches independently. If daily drawdown or losing streak thresholds are approached, it can block the second tranche even if the first is profitable.

---

## Upgrade 4 — AI Engine Upgrade (The Final Stage)

This upgrade is categorically different from the first three. Upgrades 1–3 improve the system's **rules and filters**. Upgrade 4 adds **learned intelligence**. It should only begin after the full V6 system (Upgrades 1–3) has been validated on live paper trading.

### Why XGBoost Alone Has a Ceiling

XGBoost processes each candle as an independent row of numbers. It has no temporal awareness. It cannot learn that "three consecutive failed tests of a level, with declining volume each time, is a setup." It cannot understand that the current candle is candle 8 of a 12-candle consolidation before an expansion. It sees numbers, not patterns over time.

This is a real limitation. The solution is not to replace XGBoost — it is to add a layer that addresses what XGBoost cannot do.

### The Recommended Architecture: Two-Layer Intelligence

**Layer 1 (Upgraded Inputs — Implement First)**

Before changing any model architecture, dramatically improve the data fed to XGBoost. This is the lowest-cost, highest-return AI upgrade.

- **Orderbook Imbalance:** Integrate Level 2 orderbook data from the broker API. Compute the ratio of bid liquidity to ask liquidity at the top 5 price levels. A heavily imbalanced book (e.g., 3:1 bid-to-ask) ahead of a long signal is genuine predictive alpha. This is the single best feature addition available.
- **Liquidation Data:** For crypto, Binance exposes open interest and estimated liquidation levels via its futures API. A long signal into a zone dense with short liquidations is a significantly higher-probability setup than the same signal in a neutral zone.
- **Rolling Candle Features:** Add sequential features that give XGBoost a limited form of memory — e.g., "number of consecutive bearish candles before this one," "distance from price to VWAP over last 20 candles," "ATR trend over last 10 candles."

This alone can improve model performance by 5–8% on the right setups, with zero architectural change.

**Layer 2 (Reinforcement Learning Meta-Controller — The True Intelligence)**

This is the architecture that turns Apex from a rule-based filter into a system that genuinely learns to trade.

A Reinforcement Learning agent (specifically a PPO agent — Proximal Policy Optimization) is trained not to predict price direction, but to learn the **optimal policy** for managing the entire system: when to enter, when to hold, when to add a tranche, when to sit out entirely.

```
State Space (what the agent sees):
  - All current Council votes and weights
  - Current ML probability output
  - Oracle HTF context flags
  - Open position status (in trade / flat, current PnL, trailing stop distance)
  - Portfolio-level metrics (daily PnL, drawdown, win streak / loss streak)
  - Time-of-day session flags

Action Space (what the agent can do):
  - ENTER (full size)
  - ENTER (half size, cautious)
  - HOLD / SKIP (do not trade this signal)
  - EXIT (close position early)
  - ADD_TRANCHE (pyramid)

Reward Function (what it optimizes for):
  - Primary: Sharpe-ratio-weighted PnL per episode
  - Penalty: Maximum drawdown during episode
  - Penalty: Excessive trade frequency (overtrading)
  - Penalty: Entering and immediately exiting (churning)
```

The agent trains inside a simulation environment (an OpenAI Gym-compatible wrapper around the existing engine) across thousands of historical episodes. It learns — from experience, not from rules — that sitting out during choppy regimes, pyramiding during strong structural trends, and protecting capital after losing streaks produces the best risk-adjusted returns.

**Hardware Feasibility (Ryzen 5 5600H, 8GB RAM):**
PPO on a vector state space (not pixels, not images) is CPU-friendly. Training 500,000–1,000,000 timesteps on `Stable-Baselines3` will complete in approximately 2–4 hours on the existing CPU. This is entirely local and requires no cloud compute. A full retraining can be scheduled weekly or monthly.

**The LLM Pre-Session Bias Setter (Complementary, Not Critical)**

A local LLM (Ollama + Mistral 7B, approximately 4GB RAM) runs once at market open and generates a macro bias statement. This is fed as a contextual flag into the Regime Advisor's weight adjustment, not as a per-trade vote. Latency is irrelevant since it runs only at session start.

```
Session Brief Format:
  Asset: BTC/USDT
  HTF Bias: [BEARISH / BULLISH / NEUTRAL]
  Confidence: [HIGH / MEDIUM / LOW]
  Key Level to Watch: $X
  Implication: Reduce long signal weight by 20% for session
```

This adds genuine macro context to what is otherwise a purely technical system.

---

## Complete Implementation Order

| Phase | Upgrade | Prerequisite | Estimated Complexity |
|---|---|---|---|
| **Phase 1** | Orderbook + Liquidation Features | None | Low |
| **Phase 2** | Oracle (HTF SMC Memory) | Phase 1 validated | Medium |
| **Phase 3** | Structural Trailing Stop | Phase 2 running | Medium |
| **Phase 4** | Pyramiding Tranches | Phase 3 live 30 days, positive Sharpe | High |
| **Phase 5** | RL PPO Meta-Controller | Phases 1–4 stable | Very High |
| **Phase 6** | LLM Pre-Session Bias | Phase 5 (or independent) | Low |

---

## Non-Negotiable System Rules (Unchanged Through All Upgrades)

These rules from V5 are not upgraded — they are permanent:

1. **Risk Advisor VETO is absolute.** No upgrade, including the RL agent, can override it. The VETO is hardcoded at the execution layer, below the Council.
2. **Daily loss limit is sacred.** If the limit is hit, the engine stops for the session. The RL agent does not get to "try one more trade."
3. **Loss cooldown remains active.** This is not superstition. It is protection against revenge-trading behavior that an RL agent could theoretically learn if the penalty function is miscalibrated.
4. **Paper trading before live capital.** Every phase must run on PaperBroker for a minimum of 20 trading days before any capital is committed.

---

## Closing Note

The foundation is already strong. The edge already exists in V5. The purpose of this roadmap is not to add complexity for its own sake — it is to give the engine the structural memory, the patience to hold runners, the intelligence to compound into trends, and ultimately, the learned policy that no rule-based system can match.

Build in order. Validate before proceeding. The system that goes live on real capital should be a V6 that has earned its promotion from every prior phase.

---
*Apex Intelligence Engine — V6 Master Roadmap*
*Architecture Review & Implementation Plan*

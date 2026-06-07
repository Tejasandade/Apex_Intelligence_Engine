"""
Apex Intelligence Engine V5 — The Council
============================================
The consensus engine that aggregates advisor votes into a final decision.

Think of this as a board meeting:
1. The model proposes a trade (BUY/SELL at price X)
2. Each advisor reviews the proposal from their perspective
3. They vote: APPROVE, REJECT, or ABSTAIN
4. The Council tallies votes using weighted consensus
5. Trade only fires if consensus threshold is met

Key design decisions:
- Any advisor with VETO power (Risk) can block any trade alone
- ABSTAIN votes don't count for or against — they reduce confidence
- Minimum 3/5 advisors must vote (not abstain) for a decision
- Consensus threshold is configurable (default: 60% weighted approval)

The explanation system generates human-readable logs of WHY
each decision was made — critical for debugging bad trades.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.council.base import (
    BaseAdvisor,
    AdvisorVote,
    CouncilDecision,
    Vote,
)
from src.council.advisors import (
    MomentumAdvisor,
    StructureAdvisor,
    VolumeAdvisor,
    RegimeAdvisor,
    RiskAdvisor,
    SessionAdvisor,
)
from src.core.logging import get_logger

logger = get_logger("apex.council")


class Council:
    """
    The multi-advisor consensus engine.

    Aggregates votes from all advisors and produces a final
    APPROVE/REJECT decision with full explanation.
    """

    def __init__(
        self,
        consensus_threshold: float = 0.50,
        min_voting_advisors: int = 3,
        enable_veto: bool = True,
        scalp_mode: bool = False,
    ):
        """
        Args:
            consensus_threshold: Min weighted approval ratio to approve [0, 1].
            min_voting_advisors: Min advisors that must vote (not abstain).
            enable_veto: If True, Risk advisor REJECT blocks everything.
            scalp_mode: If True, lowers thresholds and bypasses strict trend/volume constraints.
        """
        self.scalp_mode = scalp_mode
        self.consensus_threshold = 0.40 if scalp_mode else consensus_threshold
        self.min_voting_advisors = 2 if scalp_mode else min_voting_advisors
        self.enable_veto = enable_veto

        # Advisors (initialized with default weights)
        self.advisors: list[BaseAdvisor] = []
        self._risk_advisor: RiskAdvisor | None = None

        # Stats
        self._decisions_made = 0
        self._approved = 0
        self._rejected = 0

    def setup_default_advisors(self) -> None:
        """Initialize the standard 5-advisor council."""
        self._risk_advisor = RiskAdvisor(weight=0.0)

        self.advisors = [
            MomentumAdvisor(weight=1.5),
            StructureAdvisor(weight=1.2),
            VolumeAdvisor(weight=1.0),
            RegimeAdvisor(weight=1.3),
            SessionAdvisor(weight=1.0),
            self._risk_advisor,
        ]

        logger.info(
            "council_initialized",
            advisors=[a.name for a in self.advisors],
            weights=[a.weight for a in self.advisors],
            threshold=self.consensus_threshold,
        )

    def setup_india_advisors(self, broker: Any = None) -> None:
        """Initialize the India-specific council."""
        from src.council.advisors_india import IndiaSessionAdvisor, IndiaVIXAdvisor
        
        self._risk_advisor = RiskAdvisor(weight=2.0)

        self.advisors = [
            MomentumAdvisor(weight=1.5),
            StructureAdvisor(weight=1.2),
            RegimeAdvisor(weight=1.3),
            IndiaSessionAdvisor(weight=1.5),
            IndiaVIXAdvisor(broker=broker, weight=1.2),
            self._risk_advisor,
        ]

        logger.info(
            "india_council_initialized",
            advisors=[a.name for a in self.advisors],
            weights=[a.weight for a in self.advisors],
            threshold=self.consensus_threshold,
        )

    def add_advisor(self, advisor: BaseAdvisor) -> None:
        """Add a custom advisor to the council."""
        self.advisors.append(advisor)
        if isinstance(advisor, RiskAdvisor):
            self._risk_advisor = advisor

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        **kwargs: Any,
    ) -> CouncilDecision:
        """
        Run all advisors and produce a consensus decision.

        Args:
            features: Current market features (1-row DataFrame).
            direction: Proposed trade direction ("BUY" or "SELL").
            price: Current price.
            regime: Current market regime.
            atr: Current ATR value.

        Returns:
            CouncilDecision with votes, consensus score, and explanation.
        """
        votes: list[AdvisorVote] = []

        # Collect all votes
        for advisor in self.advisors:
            try:
                vote = advisor.evaluate(
                    features=features,
                    direction=direction,
                    price=price,
                    regime=regime,
                    atr=atr,
                    scalp_mode=self.scalp_mode,
                    **kwargs,
                )
                votes.append(vote)
            except Exception as e:
                logger.error(
                    "advisor_error",
                    advisor=advisor.name,
                    error=str(e),
                )
                # Advisor failure = abstain
                votes.append(AdvisorVote(
                    advisor_name=advisor.name,
                    vote=Vote.ABSTAIN,
                    conviction=0.0,
                    weight=advisor.weight,
                    reasoning=f"Error: {str(e)}",
                ))

        # ── Check for veto ───────────────────────────────────────────────────
        if self.enable_veto:
            for vote in votes:
                # Risk advisor always has veto power
                is_veto = (vote.advisor_name == "Risk" and vote.vote == Vote.REJECT)
                    
                # IndiaSessionAdvisor always has veto power
                if vote.advisor_name == "IndiaSessionAdvisor" and vote.vote == Vote.REJECT:
                    is_veto = True
                
                if is_veto:
                    decision = CouncilDecision(
                        approved=False,
                        direction=direction,
                        consensus_score=0.0,
                        votes=votes,
                        explanation=f"VETOED by {vote.advisor_name}: {vote.reasoning}",
                    )
                    self._record_decision(decision)
                    return decision

        # ── Tally weighted votes ─────────────────────────────────────────────
        weight_approve = 0.0
        weight_reject = 0.0
        weight_abstain = 0.0
        voting_count = 0

        for vote in votes:
            weighted = vote.weight * vote.conviction

            if vote.vote == Vote.APPROVE:
                weight_approve += weighted
                voting_count += 1
            elif vote.vote == Vote.REJECT:
                weight_reject += weighted
                voting_count += 1
            else:  # ABSTAIN
                weight_abstain += vote.weight

        # ── Check minimum voting requirement ─────────────────────────────────
        if voting_count < self.min_voting_advisors:
            decision = CouncilDecision(
                approved=False,
                direction=direction,
                consensus_score=0.0,
                votes=votes,
                total_weight_approve=weight_approve,
                total_weight_reject=weight_reject,
                total_weight_abstain=weight_abstain,
                explanation=(
                    f"Insufficient votes: {voting_count}/{self.min_voting_advisors} "
                    f"advisors voted (too many abstentions)"
                ),
            )
            self._record_decision(decision)
            return decision

        # ── Compute consensus ────────────────────────────────────────────────
        total_effective_weight = weight_approve + weight_reject + (weight_abstain * 0.5)
        consensus = weight_approve / total_effective_weight if total_effective_weight > 0 else 0.0
        approved = consensus >= self.consensus_threshold

        # ── Build explanation ────────────────────────────────────────────────
        explanation = self._build_explanation(votes, consensus, approved, direction)

        trade_type = "SCALP"
        if approved:
            # Handle both DataFrame and dict feature inputs
            if hasattr(features, "iloc"):
                row = features.iloc[-1]
            else:
                row = features
                
            # Dynamic Trade Type routing based on structure & sentiment
            confluence = float(row.get("structural_confluence", 0.0))
            sentiment = float(row.get("macro_sentiment_score", 0.0))
            vwap_z = float(row.get("VWAP_zscore", 0.0))
            
            # SWING if structure strongly aligns with direction
            if direction == "BUY" and (confluence > 0.5 or sentiment > 0.5):
                trade_type = "SWING"
            elif direction == "SELL" and (confluence < -0.5 or sentiment < -0.5):
                trade_type = "SWING"
            
            # Revert to SCALP if extended from VWAP (mean-reversion risk)
            if abs(vwap_z) > 2.0:
                trade_type = "SCALP"

        decision = CouncilDecision(
            approved=approved,
            direction=direction,
            consensus_score=consensus,
            trade_type=trade_type,
            votes=votes,
            total_weight_approve=weight_approve,
            total_weight_reject=weight_reject,
            total_weight_abstain=weight_abstain,
            explanation=explanation,
        )

        self._record_decision(decision)
        return decision

    def _build_explanation(
        self,
        votes: list[AdvisorVote],
        consensus: float,
        approved: bool,
        direction: str,
    ) -> str:
        """Generate a human-readable explanation of the council decision."""
        status = "APPROVED" if approved else "REJECTED"
        lines = [
            f"Council {status} {direction} (consensus: {consensus:.1%})",
            f"Threshold: {self.consensus_threshold:.1%}",
            "",
        ]

        for vote in votes:
            icon = {"APPROVE": "✅", "REJECT": "❌", "ABSTAIN": "⚪"}.get(vote.vote, "?")
            lines.append(
                f"  {icon} {vote.advisor_name:12s} "
                f"[{vote.vote:7s}] "
                f"w={vote.weight:.1f} "
                f"c={vote.conviction:.2f} "
                f"| {vote.reasoning}"
            )

        return "\n".join(lines)

    def _record_decision(self, decision: CouncilDecision) -> None:
        """Log the decision and update stats."""
        self._decisions_made += 1
        if decision.approved:
            self._approved += 1
        else:
            self._rejected += 1

        logger.info(
            "council_decision",
            direction=decision.direction,
            approved=decision.approved,
            consensus=f"{decision.consensus_score:.4f}",
            summary=decision.vote_summary,
            total_decisions=self._decisions_made,
        )

        # Log full explanation at debug level
        logger.debug("council_explanation", explanation=decision.explanation)

    def update_risk_state(
        self,
        drawdown_pct: float = 0.0,
        daily_pnl: float = 0.0,
        recent_losses: int = 0,
        open_positions: int = 0,
    ) -> None:
        """Update the Risk advisor's state."""
        if self._risk_advisor:
            self._risk_advisor.update_state(
                drawdown_pct=drawdown_pct,
                daily_pnl=daily_pnl,
                recent_losses=recent_losses,
                open_positions=open_positions,
            )

    def get_stats(self) -> dict[str, Any]:
        """Return council statistics."""
        return {
            "decisions_made": self._decisions_made,
            "approved": self._approved,
            "rejected": self._rejected,
            "approval_rate": self._approved / max(self._decisions_made, 1),
            "advisors": [
                {
                    "name": a.name,
                    "weight": a.weight,
                    "approval_rate": a.approval_rate,
                    "votes_cast": a._vote_count,
                }
                for a in self.advisors
            ],
        }

"""
Apex Intelligence Engine V5 — Council Base Classes
=====================================================
The Council is a multi-advisor signal confirmation system.
Think of it as a board of directors — each advisor is an expert
in a specific domain (momentum, structure, volume, risk).

A trade only fires when enough advisors agree.

Architecture:
    Signal Engine → Council → [Advisor 1, 2, 3, 4, 5] → Consensus → Final Decision

Each advisor:
    1. Receives the current market state (features, regime, price)
    2. Votes: APPROVE, REJECT, or ABSTAIN
    3. Provides conviction strength [0, 1]
    4. Explains WHY it voted that way (human-readable)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Vote(str, Enum):
    """An advisor's vote on a proposed trade."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


@dataclass
class AdvisorVote:
    """A single advisor's vote with reasoning."""

    advisor_name: str
    vote: Vote
    conviction: float  # 0.0 to 1.0 — how strongly they feel
    weight: float  # This advisor's voting weight
    reasoning: str  # Human-readable explanation
    details: dict[str, Any] = field(default_factory=dict)  # Raw data


@dataclass
class CouncilDecision:
    """The council's final decision after all advisors vote."""

    approved: bool
    direction: str  # BUY / SELL
    consensus_score: float  # Weighted approval ratio [0, 1]
    trade_type: str = "SWING"  # SWING or SCALP
    votes: list[AdvisorVote] = field(default_factory=list)
    total_weight_approve: float = 0.0
    total_weight_reject: float = 0.0
    total_weight_abstain: float = 0.0
    explanation: str = ""  # Summary of why the decision was made

    @property
    def approval_ratio(self) -> float:
        """Fraction of voting weight that approved."""
        total = self.total_weight_approve + self.total_weight_reject
        if total == 0:
            return 0.0
        return self.total_weight_approve / total

    @property
    def vote_summary(self) -> str:
        """Quick summary: '4/5 advisors approved'."""
        approves = sum(1 for v in self.votes if v.vote == Vote.APPROVE)
        rejects = sum(1 for v in self.votes if v.vote == Vote.REJECT)
        abstains = sum(1 for v in self.votes if v.vote == Vote.ABSTAIN)
        return f"{approves}A/{rejects}R/{abstains}S"


class BaseAdvisor(ABC):
    """
    Abstract base class for all council advisors.

    Each advisor analyzes the market from a specific perspective
    and votes on whether a proposed trade should be taken.
    """

    def __init__(self, name: str, weight: float = 1.0):
        """
        Args:
            name: Human-readable advisor name.
            weight: Voting weight (higher = more influence).
        """
        self.name = name
        self.weight = weight
        self._vote_count = 0
        self._approve_count = 0
        self._reject_count = 0

    @abstractmethod
    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        scalp_mode: bool = False,
        **kwargs: Any,
    ) -> AdvisorVote:
        """
        Evaluate a proposed trade and cast a vote.

        Args:
            features: Current feature values (1-row DataFrame).
            direction: Proposed direction ("BUY" or "SELL").
            price: Current price.
            regime: Current market regime.
            atr: Current ATR value.

        Returns:
            AdvisorVote with vote, conviction, and reasoning.
        """
        ...

    def _make_vote(
        self,
        vote: Vote,
        conviction: float,
        reasoning: str,
        details: dict[str, Any] | None = None,
    ) -> AdvisorVote:
        """Helper to create a vote and update stats."""
        self._vote_count += 1
        if vote == Vote.APPROVE:
            self._approve_count += 1
        elif vote == Vote.REJECT:
            self._reject_count += 1

        return AdvisorVote(
            advisor_name=self.name,
            vote=vote,
            conviction=min(max(conviction, 0.0), 1.0),
            weight=self.weight,
            reasoning=reasoning,
            details=details or {},
        )

    @property
    def approval_rate(self) -> float:
        if self._vote_count == 0:
            return 0.0
        return self._approve_count / self._vote_count

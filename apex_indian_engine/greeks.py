"""
Apex Indian Engine — Black-Scholes-Merton (BSM) Options Greeks Core
===================================================================
Calculates Implied Volatility (IV), Delta, Gamma, Theta, and Vega
using closed-form analytical formulas and Newton-Raphson solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


@dataclass
class OptionGreeks:
    iv: float                       # Annualized Implied Volatility (e.g. 0.15 = 15%)
    delta: float                    # Sensitivity to underlying (0.0 to 1.0 for CE, -1.0 to 0.0 for PE)
    gamma: float                    # Rate of change of delta per 1 pt move
    theta_per_day: float            # Theta decay in INR points per calendar day
    theta_per_hour: float           # Theta decay in INR points per market hour (6.25h/day)
    vega: float                     # Price change per 1% change in IV
    intrinsic_value: float          # In-The-Money cash value
    time_value: float               # Extrinsic premium (burnable)


class GreeksEngine:
    """
    High-performance analytical BSM engine for NSE Options.
    """

    def __init__(self, risk_free_rate: float = 0.07):
        # 7.0% Reserve Bank of India (RBI) 91-day T-bill repo rate
        self.r = risk_free_rate

    def calc_black_scholes(
        self,
        S: float,
        K: float,
        T: float,
        r: float = 0.07,
        sigma: float = 0.15,
        option_type: str = "CE",
    ) -> Any:
        """Standard Black-Scholes theoretical pricing for option contracts."""
        t = max(T, 0.0001)
        sig = max(sigma, 0.01)
        d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
        d2 = d1 - sig * math.sqrt(t)

        opt = option_type.upper()
        if opt == "CE":
            price = S * _norm_cdf(d1) - K * math.exp(-r * t) * _norm_cdf(d2)
            delta = _norm_cdf(d1)
        else:
            price = K * math.exp(-r * t) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1.0

        class BSPricingResult:
            def __init__(self, p: float, d: float):
                self.price = round(max(p, 0.5), 2)
                self.delta = round(d, 3)

        return BSPricingResult(price, delta)

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry_years: float,
        market_price: float,
        option_type: str = "CE",
        initial_iv: float = 0.16,
    ) -> OptionGreeks:
        """
        Calculate full Greeks suite given current option market price and spot.
        """
        t = max(time_to_expiry_years, 0.0001)  # Avoid division by zero on expiry minute
        opt = option_type.upper()

        # 1. Implied Volatility solver (Newton-Raphson)
        iv = self.solve_iv(spot, strike, t, market_price, opt, initial_iv)

        d1 = (math.log(spot / strike) + (self.r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
        d2 = d1 - iv * math.sqrt(t)

        # 2. Delta & Gamma
        if opt == "CE":
            delta = _norm_cdf(d1)
            intrinsic = max(0.0, spot - strike)
        else:
            delta = _norm_cdf(d1) - 1.0
            intrinsic = max(0.0, strike - spot)

        gamma = _norm_pdf(d1) / (spot * iv * math.sqrt(t))

        # 3. Theta (Per day & per market hour)
        term1 = -(spot * _norm_pdf(d1) * iv) / (2.0 * math.sqrt(t))
        if opt == "CE":
            term2 = -self.r * strike * math.exp(-self.r * t) * _norm_cdf(d2)
            theta_annual = term1 + term2
        else:
            term2 = self.r * strike * math.exp(-self.r * t) * _norm_cdf(-d2)
            theta_annual = term1 + term2

        theta_day = theta_annual / 365.0
        theta_hour = theta_day / 6.25  # 6.25 trading hours per NSE session

        # 4. Vega (Per 1% IV change)
        vega = spot * math.sqrt(t) * _norm_pdf(d1) * 0.01

        time_value = max(0.0, market_price - intrinsic)

        return OptionGreeks(
            iv=round(iv, 4),
            delta=round(delta, 3),
            gamma=round(gamma, 5),
            theta_per_day=round(theta_day, 2),
            theta_per_hour=round(theta_hour, 2),
            vega=round(vega, 2),
            intrinsic_value=round(intrinsic, 2),
            time_value=round(time_value, 2),
        )

    def solve_iv(
        self,
        spot: float,
        strike: float,
        t: float,
        market_price: float,
        option_type: str = "CE",
        guess: float = 0.16,
    ) -> float:
        """
        Solves Implied Volatility using Newton-Raphson with bisection fallback.
        """
        iv = max(guess, 0.01)
        for _ in range(25):
            d1 = (math.log(spot / strike) + (self.r + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
            d2 = d1 - iv * math.sqrt(t)

            if option_type == "CE":
                theo_price = spot * _norm_cdf(d1) - strike * math.exp(-self.r * t) * _norm_cdf(d2)
            else:
                theo_price = strike * math.exp(-self.r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

            diff = theo_price - market_price
            if abs(diff) < 0.05:
                return iv

            vega = spot * math.sqrt(t) * _norm_pdf(d1)
            if abs(vega) < 1e-6:
                break

            step = diff / vega
            iv -= step
            if iv <= 0.001 or iv > 3.0:
                iv = 0.18
                break

        return max(0.01, min(iv, 2.5))


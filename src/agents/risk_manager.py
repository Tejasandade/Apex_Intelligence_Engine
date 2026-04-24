from loguru import logger

class RiskManager:
    """
    The Executive Risk Manager module.
    Responsible for capital allocation, enforcing drawdown limits,
    and making final trade execution decisions based on Quant Agent signals.
    """
    
    def __init__(self, total_capital: float = 10000.0, max_risk_per_trade: float = 0.02, daily_drawdown_limit: float = 0.05):
        self.total_capital = total_capital
        self.starting_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.daily_drawdown_limit = daily_drawdown_limit
        self.circuit_breaker_tripped = False
        logger.info(f"Initialized RiskManager with ${self.total_capital} capital and {self.max_risk_per_trade*100}% max risk per trade.")
        
    def check_circuit_breaker(self, current_balance: float):
        """
        Calculates total percentage lost from starting_capital.
        Trips the circuit breaker if the daily drawdown limit is exceeded.
        """
        self.total_capital = current_balance
        if self.circuit_breaker_tripped:
            return
            
        drawdown = (self.starting_capital - current_balance) / self.starting_capital
        if drawdown >= self.daily_drawdown_limit:
            self.circuit_breaker_tripped = True
            logger.warning('CRITICAL: Daily drawdown limit reached. Trading halted.')

    def evaluate_trade_signal(self, symbol: str, bullish_probability: float) -> dict:
        """
        Evaluates a signal from the Quant Model. 
        Determines whether a trade should be taken and in what direction.
        """
        if self.circuit_breaker_tripped:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "probability": bullish_probability
            }
            
        logger.debug(f"Evaluating signal for {symbol}: Bullish Prob = {bullish_probability}")
        
        # Directional thresholds — tightened for model's natural output range
        action = "HOLD"
        if bullish_probability > 0.55:
            action = "BUY"
        elif bullish_probability < 0.45:
            action = "SELL"
            
        return {
            "symbol": symbol,
            "action": action,
            "probability": bullish_probability
        }
        
    def calculate_position_size(self, symbol: str, action: str, confidence: float, reward_risk_ratio: float = 1.5, kelly_fraction: float = 0.5) -> float:
        """
        Calculates the exact position size based on current capital, risk limits, 
        and signal confidence using a fractional Kelly criterion.
        """
        if action == "HOLD":
            return 0.0
            
        logger.debug(f"Calculating position size for {action} on {symbol} (Confidence: {confidence})")
        
        # Kelly Criterion: p - ((1 - p) / reward_risk_ratio)
        probability = confidence
        kelly_pct = probability - ((1.0 - probability) / reward_risk_ratio)
        
        # Negative edge protection
        if kelly_pct <= 0:
            logger.info("Kelly percentage is negative (no statistical edge). Aborting trade.")
            return 0.0
            
        # Calculate raw allocation using fractional Kelly
        raw_allocation = self.total_capital * (kelly_pct * kelly_fraction)
        
        # Crucial Safety Cap: Do not exceed max_risk_per_trade
        max_allowed_risk = self.total_capital * self.max_risk_per_trade
        final_allocation = min(raw_allocation, max_allowed_risk)
        
        logger.debug(f"Kelly %: {kelly_pct:.2f} | Raw Allocation: ${raw_allocation:.2f} | Capped Allocation: ${final_allocation:.2f}")
        
        return round(final_allocation, 2)

if __name__ == "__main__":
    # Quick test of the dynamic sizing and circuit breaker
    risk_manager = RiskManager(total_capital=10000.0, max_risk_per_trade=0.05, daily_drawdown_limit=0.05)
    
    # Simulate a loss that exceeds the 5% drawdown limit
    risk_manager.check_circuit_breaker(9400.0)
    
    signal = risk_manager.evaluate_trade_signal("BTCUSDT", 0.85)
    size = risk_manager.calculate_position_size(
        signal["symbol"], 
        signal["action"], 
        signal["probability"],
        reward_risk_ratio=1.5,
        kelly_fraction=0.5
    )
    
    logger.info(f"Final Decision: {signal['action']} {signal['symbol']} with size ${size}")

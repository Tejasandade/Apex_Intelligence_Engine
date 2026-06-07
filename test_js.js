const state = { currency: '₹' };
const data = {
    direction: "SELL",
    symbol: "banknifty",
    price: 54239.2,
    conviction: 0.3179,
    council_score: 0.7081,
    blocked: false,
    reason: "TRENDING regime",
    entry_price: 54239.2,
    stop_loss: 54408.21,
    take_profit: 53901.18
};

try {
    const s1 = `<span>Price: ${state.currency || '$'}${(data.price || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>`;
    const s2 = `<div class="position-card__pnl" id="positionPnl">${state.currency || '$'}0.00</div>`;
    console.log("SUCCESS:", s1);
    console.log("SUCCESS:", s2);
} catch (e) {
    console.log("ERROR:", e);
}

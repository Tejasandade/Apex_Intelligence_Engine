function formatCurrency(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return '--'
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(Number(value))
}

function ActiveTrades({ trades, closingOrderId, onCloseTrade, onClearStale }) {
  const visibleTrades = trades.filter(t => t.broker_status !== 'CORRUPTED')

  return (
    <section className="panel">
      <div className="panel-title-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="panel-title">Active Trades</div>
        <button className="execute-button" style={{ padding: '4px 10px', fontSize: '0.8rem' }} onClick={onClearStale}>
          Clear All Stale
        </button>
      </div>
      <div className="active-trades-list">
        {visibleTrades.length === 0 ? (
          <div className="trade-item trade-item--empty">No open positions in the OMS blotter.</div>
        ) : (
          trades.map((trade) => (
            <article key={trade.order_id} className="active-trade-card">
              <div className="active-trade-card__header">
                <div>
                  <div className="trade-signal">{trade.symbol}</div>
                  <div className="trade-meta">{trade.side} / {trade.status}</div>
                  <div className="trade-meta">{trade.broker_status}{trade.exit_reason ? ` / ${trade.exit_reason}` : ''}</div>
                </div>
                <button
                  type="button"
                  className="close-trade-button"
                  disabled={closingOrderId === trade.order_id || trade.status === 'SQUARING OFF'}
                  onClick={() => onCloseTrade(trade.order_id)}
                >
                  {trade.status === 'SQUARING OFF'
                    ? 'SQUARING OFF'
                    : closingOrderId === trade.order_id
                      ? 'Closing...'
                      : 'Close Trade'}
                </button>
              </div>

              <div className="active-trade-grid">
                <div className="metric-item">
                  <span>Entry Price</span>
                  <strong>{formatCurrency(trade.entry_price)}</strong>
                </div>
                <div className="metric-item">
                  <span>Current Price</span>
                  <strong>{formatCurrency(trade.current_price)}</strong>
                </div>
                <div className="metric-item">
                  <span>Live PnL</span>
                  <strong className={trade.pnl_pct >= 0 ? 'text-positive' : 'text-negative'}>
                    {trade.pnl_pct >= 0 ? '+' : ''}{Number(trade.pnl_pct ?? 0).toFixed(2)}%
                  </strong>
                </div>
                <div className="metric-item">
                  <span>Trailing Stop</span>
                  <strong>{formatCurrency(trade.trailing_stop_level)}</strong>
                </div>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  )
}

export default ActiveTrades

import React, { useState } from 'react'

const API_URL = 'http://127.0.0.1:8080'

function formatMarketCurrency(value, activeTab) {
  if (value == null || Number.isNaN(Number(value))) return '--'
  if (activeTab === 'INDIA') {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency', currency: 'INR', maximumFractionDigits: 2,
    }).format(Number(value))
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency', currency: 'USD', maximumFractionDigits: 2,
  }).format(Number(value))
}

// ── Inline regime badge (same design language as SmartOrderCard) ─────────
function RegimeChip({ regime }) {
  if (!regime || regime === 'Unknown') return null
  const isTrending = regime === 'Trending'
  const color  = isTrending ? '#22d3ee' : '#f59e0b'
  const bg     = isTrending ? 'rgba(34,211,238,0.10)' : 'rgba(245,158,11,0.10)'
  const border = isTrending ? 'rgba(34,211,238,0.30)' : 'rgba(245,158,11,0.30)'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '4px',
      padding: '2px 8px', borderRadius: '20px',
      background: bg, border: `1px solid ${border}`,
      color, fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.5px',
    }}>
      {isTrending ? '📈' : '🌀'} {regime}
    </span>
  )
}

function HighConvictionSignals({ signals, activeTab = 'CRYPTO' }) {
  const [expandedId, setExpandedId] = useState(null)
  const [executingId, setExecutingId] = useState(null)
  const [cancellingId, setCancellingId] = useState(null)

  const toggleExpand = (orderId) => {
    setExpandedId(expandedId === orderId ? null : orderId)
  }

  const executeSignal = async (e, signal) => {
    e.stopPropagation()
    setExecutingId(signal.order_id)
    try {
      await fetch(`${API_URL}/api/orders/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          order_id: signal.order_id,
          symbol: signal.symbol,
          reason: 'manual_override',
        }),
      })
    } catch (err) {
      console.error('Execute failed:', err)
    } finally {
      setExecutingId(null)
    }
  }

  const cancelSignal = async (e, orderId) => {
    e.stopPropagation()
    setCancellingId(orderId)
    try {
      await fetch(`${API_URL}/api/orders/close/${orderId}`, { method: 'POST' })
    } catch (err) {
      console.error('Cancel failed:', err)
    } finally {
      setCancellingId(null)
    }
  }

  return (
    <section className="panel">
      <div className="panel-title-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div className="panel-title">High-Conviction Signals</div>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', letterSpacing: '1px' }}>
          {signals?.length ?? 0} ACTIVE
        </div>
      </div>
      <div className="active-trades-list">
        {!signals || signals.length === 0 ? (
          <div className="trade-item trade-item--empty">No high-conviction signals generated yet. Engine is monitoring...</div>
        ) : (
          signals.map((signal) => {
            const isExpanded   = expandedId === signal.order_id
            const isBuy        = signal.signal === 'BUY'
            const signalColor  = isBuy ? 'var(--bullish)' : 'var(--bearish)'
            const isExecuting  = executingId === signal.order_id
            const isCancelling = cancellingId === signal.order_id
            const ep           = signal.execution_plan

            // Regime data lives on signal_analysis (from TradeTicket.features is not the right path;
            // the backend places it on execution_plan's parent signal_analysis via the card broadcast)
            // For signals, we get it via the features dict if available
            const analysis  = signal.features?.signal_analysis ?? signal.signal_analysis ?? null
            const regime    = analysis?.regime_classification ?? null
            const atrDist   = analysis?.atr_distance ?? null

            return (
              <article
                key={signal.order_id}
                className="active-trade-card"
                style={{ cursor: 'pointer', transition: 'all 0.2s ease', borderLeft: `3px solid ${signalColor}` }}
                onClick={() => toggleExpand(signal.order_id)}
              >
                {/* ── Header row ── */}
                <div className="active-trade-card__header" style={{ alignItems: 'flex-start' }}>
                  <div>
                    <div className="trade-signal" style={{ color: signalColor, fontWeight: 700 }}>
                      {signal.symbol} {signal.signal}
                    </div>
                    <div className="trade-meta">
                      {((signal.probability ?? 0.5) * 100).toFixed(2)}% EDGE | {signal.provider}
                    </div>
                    {/* Regime chip inline */}
                    {regime && (
                      <div style={{ marginTop: '5px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <RegimeChip regime={regime} />
                        {atrDist != null && atrDist > 0 && (
                          <span style={{
                            fontSize: '0.62rem', color: 'var(--text-dim)',
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid var(--border)',
                            padding: '2px 7px', borderRadius: '12px',
                          }}>
                            ATR {atrDist.toFixed(2)}%
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                  <div style={{ textAlign: 'right', fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                    <div>{signal.date}</div>
                    <div style={{ color: 'var(--text-dimmer)' }}>{signal.time}</div>
                  </div>
                </div>

                {/* ── Key metrics grid ── */}
                <div className="active-trade-grid" style={{ marginTop: '10px' }}>
                  <div className="metric-item">
                    <span>Entry</span>
                    <strong>{formatMarketCurrency(ep?.entry_price, activeTab)}</strong>
                  </div>
                  <div className="metric-item">
                    <span>Target</span>
                    <strong style={{ color: 'var(--bullish)' }}>{formatMarketCurrency(ep?.take_profit, activeTab)}</strong>
                  </div>
                  <div className="metric-item">
                    <span>Stop</span>
                    <strong style={{ color: 'var(--bearish)' }}>{formatMarketCurrency(ep?.stop_loss, activeTab)}</strong>
                  </div>
                  <div className="metric-item">
                    <span>R/R</span>
                    <strong>{ep?.risk_reward?.toFixed(2) ?? '--'}</strong>
                  </div>
                </div>

                {/* ── Execute & Cancel ── */}
                <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }} onClick={e => e.stopPropagation()}>
                  <button
                    onClick={(e) => executeSignal(e, signal)}
                    disabled={isExecuting || isCancelling}
                    style={{
                      flex: 1, padding: '7px 12px', borderRadius: '4px',
                      background: signalColor, border: 'none', color: '#fff',
                      fontWeight: 700, fontSize: '0.75rem', cursor: 'pointer',
                      opacity: isExecuting ? 0.6 : 1,
                    }}
                  >
                    {isExecuting ? 'Routing...' : `⚡ Execute ${signal.signal}`}
                  </button>
                  <button
                    onClick={(e) => cancelSignal(e, signal.order_id)}
                    disabled={isExecuting || isCancelling}
                    style={{
                      padding: '7px 12px', borderRadius: '4px',
                      background: 'transparent', border: '1px solid var(--border)',
                      color: 'var(--text-dim)', fontWeight: 600, fontSize: '0.75rem', cursor: 'pointer',
                    }}
                  >
                    {isCancelling ? '...' : '✕ Dismiss'}
                  </button>
                </div>

                {/* ── Expandable analysis ── */}
                {isExpanded && (
                  <div style={{
                    marginTop: '14px', padding: '12px', background: 'var(--bg)',
                    borderRadius: '6px', fontSize: '0.75rem', color: 'var(--text-dim)',
                    lineHeight: '1.6', border: '1px solid var(--border)'
                  }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '10px' }}>
                      <div>
                        <div style={{ color: 'var(--text-dimmer)', fontSize: '0.65rem', marginBottom: '2px' }}>ALLOCATION</div>
                        <div style={{ color: 'var(--text)', fontWeight: 600 }}>{formatMarketCurrency(ep?.allocation, activeTab)}</div>
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-dimmer)', fontSize: '0.65rem', marginBottom: '2px' }}>EST. PnL</div>
                        <div style={{ color: 'var(--bullish)', fontWeight: 600 }}>{formatMarketCurrency(ep?.est_pnl, activeTab)}</div>
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-dimmer)', fontSize: '0.65rem', marginBottom: '2px' }}>MAX LOSS</div>
                        <div style={{ color: 'var(--bearish)', fontWeight: 600 }}>{formatMarketCurrency(ep?.max_loss, activeTab)}</div>
                      </div>
                      <div>
                        <div style={{ color: 'var(--text-dimmer)', fontSize: '0.65rem', marginBottom: '2px' }}>SENTIMENT</div>
                        <div style={{ color: 'var(--text)', fontWeight: 600 }}>{((signal.sentiment ?? 0.5) * 100).toFixed(1)}%</div>
                      </div>
                      {/* ── Regime + ATR in expanded view ── */}
                      {regime && (
                        <div style={{ gridColumn: '1 / -1' }}>
                          <div style={{ color: 'var(--text-dimmer)', fontSize: '0.65rem', marginBottom: '4px' }}>MARKET REGIME</div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <RegimeChip regime={regime} />
                            {atrDist != null && atrDist > 0 && (
                              <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                                ATR Distance: <strong style={{ color: 'var(--text)' }}>{atrDist.toFixed(2)}%</strong>
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    {analysis?.technical_summary && (
                      <div style={{ marginBottom: '6px', paddingTop: '8px', borderTop: '1px solid var(--border)' }}>
                        <span style={{ color: 'var(--text)', fontWeight: 600 }}>📊 Technical: </span>
                        {analysis.technical_summary}
                      </div>
                    )}
                    {analysis?.sentiment_summary && (
                      <div>
                        <span style={{ color: 'var(--text)', fontWeight: 600 }}>📰 Sentiment: </span>
                        {analysis.sentiment_summary}
                      </div>
                    )}
                  </div>
                )}
              </article>
            )
          })
        )}
      </div>
    </section>
  )
}

export default HighConvictionSignals

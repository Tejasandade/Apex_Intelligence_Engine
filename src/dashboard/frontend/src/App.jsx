import { useEffect, useRef, useState } from 'react'
import ActiveTrades from './components/ActiveTrades.jsx'
import SmartOrderCard from './components/SmartOrderCard.jsx'
import TerminalConsole from './components/TerminalConsole.jsx'

const API_URL = 'http://127.0.0.1:8080'
const WS_URL = 'ws://127.0.0.1:8080/ws/live'

// ---------------------------------------------------------------------------
// TradingView Advanced Real-Time Chart — clean embed URL
// ---------------------------------------------------------------------------
function TradingViewChart({ symbol = 'BINANCE:BTCUSDTPERP' }) {
  const src =
    `https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}` +
    `&interval=1&theme=dark&style=1&locale=en` +
    `&toolbar_bg=%23141418&enable_publishing=0` +
    `&allow_symbol_change=0&save_image=0&hide_top_toolbar=0` +
    `&withdateranges=1&hide_side_toolbar=0` +
    `&container_id=tradingview_apex`

  return (
    <section className="panel tradingview-hero" aria-label="Live Price Chart">
      <div className="panel-title-row">
        <div className="panel-title">Live Chart — {symbol}</div>
        <div className="chart-badge">BINANCE · REAL-TIME</div>
      </div>
      <div className="tradingview-wrapper">
        <iframe
          id="tradingview_apex"
          title={`TradingView Advanced Chart — ${symbol}`}
          src={src}
          frameBorder="0"
          allowTransparency="true"
          scrolling="no"
          allow="autoplay"
          style={{ width: '100%', height: '100%', border: 'none', display: 'block' }}
        />
      </div>
    </section>
  )
}

function formatCurrency(value) {
  if (value == null || Number.isNaN(Number(value))) return '--'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(Number(value))
}

function App() {
  const wsRef = useRef(null)
  const [snapshot, setSnapshot] = useState(null)
  const [capitalInput, setCapitalInput] = useState('10000')
  const [tradeAllocationInput, setTradeAllocationInput] = useState('0')
  const [wsConnected, setWsConnected] = useState(false)
  const [savingCapital, setSavingCapital] = useState(false)
  const [togglingMode, setTogglingMode] = useState(false)
  const [executing, setExecuting] = useState(false)
  const [closingOrderId, setClosingOrderId] = useState(null)
  const [livePrice, setLivePrice] = useState(null)
  const [tickActive, setTickActive] = useState(false)
  const [pollFailures, setPollFailures] = useState(0)

  // 100ms Zero-Latency Price Polling from Redis Bridge
  useEffect(() => {
    let active = true
    const pollPrice = async () => {
      try {
        const res = await fetch(`${API_URL}/api/market/tick`)
        if (!res.ok) throw new Error('Bad response')
        const data = await res.json()
        if (active && data?.price > 0) {
          setLivePrice(data.price)
          setTickActive(true)
          setPollFailures(0)
          setTimeout(() => { if (active) setTickActive(false) }, 50)
        }
      } catch (err) {
        if (active) setPollFailures(prev => prev + 1)
      } finally {
        if (active) setTimeout(pollPrice, 100)
      }
    }
    pollPrice()
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true

    const loadSnapshot = async () => {
      try {
        const response = await fetch(`${API_URL}/api/dashboard/snapshot`)
        const data = await response.json()
        if (!active) return
        setSnapshot(data)
        setCapitalInput(String(data?.session?.total_capital ?? 10000))
        setTradeAllocationInput(String(data?.session?.trade_allocation ?? 0))
      } catch (error) {
        console.error('Failed to load dashboard snapshot', error)
      }
    }

    loadSnapshot()

    const connect = () => {
      const websocket = new WebSocket(WS_URL)
      wsRef.current = websocket

      websocket.onopen = () => {
        setWsConnected(true)
        websocket.send(JSON.stringify({ type: 'dashboard.subscribe' }))
      }

      websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          if (message?.payload) {
            setSnapshot(message.payload)
          }
        } catch (error) {
          console.error('Failed to parse WebSocket payload', error)
        }
      }

      websocket.onclose = () => {
        setWsConnected(false)
        if (active) window.setTimeout(connect, 3000)
      }

      websocket.onerror = () => websocket.close()
    }

    connect()

    return () => {
      active = false
      wsRef.current?.close()
    }
  }, [])

  const syncCapital = async () => {
    setSavingCapital(true)
    try {
      const response = await fetch(`${API_URL}/api/capital`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_capital: Number(capitalInput),
          trade_allocation: Number(tradeAllocationInput)
        }),
      })
      const data = await response.json()
      setCapitalInput(String(data.total_capital))
      setTradeAllocationInput(String(data.trade_allocation ?? 0))
    } catch (error) {
      console.error('Failed to sync capital', error)
    } finally {
      setSavingCapital(false)
    }
  }

  const clearStaleTrades = async () => {
    try {
      await fetch(`${API_URL}/api/orders/clear-stale`, { method: 'POST' })
    } catch (error) {
      console.error('Clear stale failed:', error)
    }
  }

  const toggleProfessionalTrader = async () => {
    setTogglingMode(true)
    try {
      await fetch(`${API_URL}/api/risk/professional-trader`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          enabled: !snapshot?.session?.professional_trader_enabled,
        }),
      })
    } catch (error) {
      console.error('Failed to toggle Professional Trader mode', error)
    } finally {
      setTogglingMode(false)
    }
  }

  const executeNow = async () => {
    setExecuting(true)
    try {
      await fetch(`${API_URL}/api/orders/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          order_id: snapshot?.smart_order_card?.order_id,
          symbol: snapshot?.market?.symbol,
          reason: 'manual_override',
        }),
      })
    } catch (error) {
      console.error('Manual execution request failed', error)
    } finally {
      setExecuting(false)
    }
  }

  const closeTrade = async (orderId) => {
    setClosingOrderId(orderId)
    try {
      await fetch(`${API_URL}/api/orders/close/${orderId}`, { method: 'POST' })
    } catch (error) {
      console.error('Close trade request failed', error)
    } finally {
      setClosingOrderId(null)
    }
  }

  const session = snapshot?.session
  const market = snapshot?.market
  const trades = snapshot?.trades ?? []
  const activeTrades = snapshot?.active_trades ?? []
  const news = snapshot?.news ?? []
  const hasActiveTrade = activeTrades.length > 0
  const activeTiers = activeTrades.map(t => t.confidence_tier || 1)

  return (
    <div className="app-shell">
      {/* ── Reconnecting Banner ──────────────────────────────────────────────── */}
      {!wsConnected && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999,
          background: 'linear-gradient(90deg, #b45309, #d97706)',
          color: '#fff', textAlign: 'center',
          padding: '8px 16px', fontSize: '0.8rem', fontWeight: 600,
          letterSpacing: '0.05em', display: 'flex', alignItems: 'center',
          justifyContent: 'center', gap: '8px',
          animation: 'pulse 1.5s ease-in-out infinite',
        }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#fff', animation: 'pulse 1s ease-in-out infinite' }} />
          RECONNECTING TO APEX ENGINE...
        </div>
      )}
      {/* ── Topbar ─────────────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="eyebrow">Phase 4</div>
          <div className="brand">Apex Command Center</div>
        </div>
        <div className="topbar-metrics">
          <div className={`status-chip ${wsConnected ? 'status-chip--live' : ''}`}>
            {wsConnected ? (
              <span className={`tick-dot ${pollFailures > 3 ? 'tick-dot--error' : (tickActive ? 'tick-dot--active' : '')}`} />
            ) : (
              <span className="status-dot" />
            )}
            {wsConnected ? 'LIVE FEED' : 'RECONNECTING'}
          </div>
          {session?.db_status === 'disconnected' && (
            <div className="status-chip" style={{ color: '#ff4d4d', borderColor: '#ff4d4d', backgroundColor: 'rgba(255, 77, 77, 0.1)' }}>
              <span className="tick-dot" style={{ backgroundColor: '#ff4d4d', boxShadow: '0 0 8px #ff4d4d' }} />
              <strong>DB OFFLINE</strong>
            </div>
          )}
          <div className="topbar-stat">
            <span>Cycle</span>
            <strong>{session?.cycle_count ?? 0}</strong>
          </div>
          <div className="topbar-stat">
            <span>Mode</span>
            <strong>{session?.execution_mode ?? 'MANUAL'}</strong>
          </div>
          <div className="topbar-stat">
            <span>Capital</span>
            <strong>{formatCurrency(session?.total_capital)}</strong>
          </div>
          <div className={`topbar-stat pnl ${(session?.sim_pnl ?? 0) >= 0 ? 'pnl--pos' : 'pnl--neg'}`}>
            <span>Session PnL</span>
            <strong>{formatCurrency(session?.sim_pnl)}</strong>
          </div>
        </div>
      </header>

      {/* ── Dashboard Grid ─────────────────────────────────────────────────── */}
      <main className="dashboard-grid">

        {/* ── LEFT SIDEBAR ─────────────────────────────────────────────────── */}
        <aside className="sidebar">

          {/* Algo-Vault Toggle */}
          <section className="panel panel--compact">
            <div className="panel-title">Algo-Vault</div>
            <div className="mode-toggle">
              <div>
                <div className="mode-label">Pro Trader</div>
                <div className="mode-copy" style={{ fontSize: '0.72rem' }}>
                  {session?.professional_trader_enabled
                    ? '⚡ Autonomous — engine armed.'
                    : 'Manual approval required.'}
                </div>
              </div>
              <button
                type="button"
                id="toggle-professional-trader"
                className={`toggle-button ${session?.professional_trader_enabled ? 'toggle-button--active' : ''}`}
                onClick={toggleProfessionalTrader}
                disabled={togglingMode}
              >
                {togglingMode ? '...' : session?.professional_trader_enabled ? 'ON' : 'OFF'}
              </button>
            </div>
          </section>

          {/* Capital Control */}
          <section className="panel panel--compact">
            <div className="panel-title">Capital Control</div>
            <div className="capital-row">
              <span className="input-label">Total Capital (USD)</span>
              <input
                id="capital-input"
                className="capital-input"
                type="number"
                value={capitalInput}
                onChange={(e) => setCapitalInput(e.target.value)}
              />
            </div>
            <div className="capital-row">
              <span className="input-label">Trade Amt (USD · 0 = Auto)</span>
              <input
                id="trade-allocation-input"
                className="capital-input"
                type="number"
                placeholder="0"
                value={tradeAllocationInput}
                onChange={(e) => setTradeAllocationInput(e.target.value)}
              />
            </div>
            <button
              id="sync-capital-button"
              type="button"
              className="action-button"
              style={{ width: '100%', marginBottom: '10px' }}
              onClick={syncCapital}
              disabled={savingCapital}
            >
              {savingCapital ? 'Saving...' : '⟳  Sync Settings'}
            </button>
            <div className="metric-list">
              <div className="metric-item">
                <span>Symbol</span>
                <strong>{market?.symbol ?? 'BTCUSDT'}</strong>
              </div>
            </div>
          </section>

          {/* Live Market Pulse */}
          <section className="panel panel--compact">
            <div className="panel-title">Market Pulse</div>
            <div className="metric-list">
              {/* Hero Price Row */}
              <div className="metric-item metric-item--price">
                <span>Live Price</span>
                <span className="price-value">{formatCurrency(livePrice ?? market?.close_price)}</span>
              </div>
              <div className="metric-item">
                <span>RSI (14)</span>
                <strong style={{
                  color: market?.rsi > 70 ? 'var(--bearish)' : market?.rsi < 30 ? 'var(--bullish)' : 'var(--text)'
                }}>
                  {market?.rsi?.toFixed(1) ?? '--'}
                  {market?.rsi > 70 ? ' ↑OB' : market?.rsi < 30 ? ' ↓OS' : ''}
                </strong>
              </div>
              <div className="metric-item">
                <span>VWAP</span>
                <strong style={{
                  color: (livePrice ?? 0) > (market?.vwap ?? 0) ? 'var(--bullish)' : 'var(--bearish)'
                }}>{formatCurrency(market?.vwap)}</strong>
              </div>
              <div className="metric-item">
                <span>ATR</span>
                <strong>{market?.atr?.toFixed(2) ?? '--'}</strong>
              </div>
              <div className="metric-item">
                <span>Signal</span>
                <strong className={`signal-text signal-text--${(market?.signal ?? 'hold').toLowerCase()}`}>
                  {market?.signal === 'BUY' ? '▲ BUY' : market?.signal === 'SELL' ? '▼ SELL' : '— HOLD'}
                </strong>
              </div>
            </div>
          </section>

          {/* Sentiment Wire */}
          <section className="panel panel--compact panel--news">
            <div className="panel-title">Sentiment Wire</div>
            <div className="news-list">
              {news.length === 0 ? (
                <div className="news-item">Scanning macro headlines...</div>
              ) : (
                news.slice(0, 8).map((headline, index) => (
                  <div key={`${index}-${headline.slice(0, 20)}`} className="news-item">
                    {headline}
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>


        {/* ── MAIN CONTENT ─────────────────────────────────────────────────── */}
        <section className="content">

          {/* ── HERO: Full-Width Chart ──────────────────────────────────────── */}
          <TradingViewChart symbol="BINANCE:BTCUSDTPERP" />

          {/* ── MID ROW: SmartOrderCard + Active Trades ─────────────────────── */}
          <div className="content-mid">
            <SmartOrderCard
              card={snapshot?.smart_order_card}
              onExecute={executeNow}
              executing={executing}
              activeTiers={activeTiers}
            />

            <div className="mid-right">
              {/* Signal Diagnostics */}
              <section className="panel panel--compact">
                <div className="panel-title">Signal Diagnostics</div>
                <div className="diag-grid">
                  <div className="diag-item">
                    <span>Bullish</span>
                    <strong className="text-bullish">
                      {((market?.probability ?? 0.5) * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div className="diag-item">
                    <span>Bearish</span>
                    <strong className="text-bearish">
                      {((1 - (market?.probability ?? 0.5)) * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div className="diag-item">
                    <span>Sentiment</span>
                    <strong>{((market?.sentiment ?? 0.5) * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="diag-item">
                    <span>Signal</span>
                    <strong className={`signal-text signal-text--${(market?.signal ?? 'hold').toLowerCase()}`}>
                      {market?.signal ?? 'HOLD'}
                    </strong>
                  </div>
                </div>
              </section>

              {/* Active Trades */}
              <ActiveTrades
                trades={activeTrades}
                closingOrderId={closingOrderId}
                onCloseTrade={closeTrade}
                onClearStale={clearStaleTrades}
              />
            </div>
          </div>

          {/* ── BOTTOM: Execution Queue ─────────────────────────────────────── */}
          <section className="panel panel--compact">
            <div className="panel-title">Execution Queue</div>
            <div className="trade-list">
              {trades.length === 0 ? (
                <div className="trade-item trade-item--empty">No staged or executed trades yet.</div>
              ) : (
                trades.map((trade) => (
                  <div key={trade.order_id} className="trade-item">
                    <div>
                      <div className="trade-signal">{trade.signal}</div>
                      <div className="trade-meta">{trade.symbol} / {trade.status}</div>
                    </div>
                    <div className="trade-side">
                      <div>{trade.time}</div>
                      <div>{trade.trigger_source}</div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* ── TERMINAL CONSOLE ──────────────────────────────────────────── */}
          <TerminalConsole />
        </section>
      </main>
    </div>
  )
}

export default App

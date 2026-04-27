import { useEffect, useRef, useState } from 'react'
import HighConvictionSignals from './components/HighConvictionSignals.jsx'
import SmartOrderCard from './components/SmartOrderCard.jsx'
import TerminalConsole from './components/TerminalConsole.jsx'

const API_URL = 'http://127.0.0.1:8080'
const WS_URL = 'ws://127.0.0.1:8080/ws/live'

// ---------------------------------------------------------------------------
// GlobalMasterView — Alpha Ranker Cross-Market Conviction Terminal
// ---------------------------------------------------------------------------
function GlobalMasterView({ snapshot }) {
  const signal = snapshot?.global_best_signal
  const isBull = signal?.direction === 'BULLISH'
  const dirColor  = isBull ? '#00e5a0' : '#ff4d6d'
  const dirBg     = isBull ? 'rgba(0,229,160,0.07)' : 'rgba(255,77,109,0.07)'
  const conviction = signal?.conviction ?? 0
  const pct        = signal ? (signal.probability * 100).toFixed(2) : null

  // Build a conviction bar width (conviction is 0–100)
  const barWidth = `${Math.min(conviction, 100)}%`

  const markets = [
    { key: 'CRYPTO', label: 'Crypto', icon: '₿' },
    { key: 'INDIA',  label: 'India NSE', icon: '₹' },
    { key: 'FOREX',  label: 'Forex', icon: '€' },
  ]

  return (
    <div style={{
      padding: '32px 40px',
      minHeight: '60vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '28px',
    }}>

      {/* ── Header ──────────────────────────────────────────────── */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          fontSize: '0.72rem', letterSpacing: '4px', color: 'var(--text-dim)',
          textTransform: 'uppercase', marginBottom: '6px',
        }}>Apex Intelligence Engine</div>
        <h2 style={{
          margin: 0, fontSize: '1.9rem', fontWeight: 700,
          background: 'linear-gradient(135deg, #a78bfa, #60a5fa)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          letterSpacing: '1px',
        }}>Global Alpha Ranker</h2>
        <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '6px' }}>
          Cross-market statistical edge — highest conviction signal wins execution
        </div>
      </div>

      {/* ── Main conviction card ──────────────────────────────────────── */}
      <div style={{
        background: 'var(--panel-bg)',
        border: `1px solid ${signal ? dirColor + '55' : 'var(--border)'}`,
        borderRadius: '16px',
        padding: '32px 40px',
        width: '100%',
        maxWidth: '620px',
        boxShadow: signal ? `0 0 32px ${dirColor}22` : 'none',
        transition: 'box-shadow 0.4s ease, border-color 0.4s ease',
        position: 'relative',
        overflow: 'hidden',
      }}>

        {/* subtle corner badge */}
        <div style={{
          position: 'absolute', top: 12, right: 16,
          fontSize: '0.65rem', letterSpacing: '3px', color: 'var(--text-dim)',
          textTransform: 'uppercase',
        }}>Live</div>

        <div style={{ fontSize: '0.72rem', letterSpacing: '3px', color: 'var(--text-dim)', textTransform: 'uppercase', marginBottom: '12px' }}>
          Highest Conviction Market
        </div>

        {signal ? (
          <>
            {/* Market name */}
            <div style={{
              fontSize: '2.8rem', fontWeight: 800,
              letterSpacing: '3px', color: 'var(--text)',
              marginBottom: '6px', fontFamily: 'monospace',
            }}>
              {signal.market ?? '—'}
            </div>

            {/* Direction + probability chip */}
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: '10px',
              background: dirBg, border: `1px solid ${dirColor}66`,
              borderRadius: '8px', padding: '8px 18px', marginBottom: '20px',
            }}>
              <span style={{ fontSize: '1.1rem' }}>{isBull ? '▲' : '▼'}</span>
              <span style={{ fontSize: '1.3rem', fontWeight: 700, color: dirColor, letterSpacing: '1px' }}>
                {signal.direction}
              </span>
              <span style={{ fontSize: '1.1rem', color: 'var(--text-dim)' }}>
                {pct}%
              </span>
            </div>

            {/* Conviction meter */}
            <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              <span>Conviction Score</span>
              <span style={{ color: 'var(--text)', fontWeight: 700 }}>{conviction} / 100</span>
            </div>
            <div style={{
              height: '8px', borderRadius: '4px',
              background: 'rgba(255,255,255,0.07)',
              overflow: 'hidden',
            }}>
              <div style={{
                height: '100%', width: barWidth,
                background: `linear-gradient(90deg, ${dirColor}88, ${dirColor})`,
                borderRadius: '4px',
                transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)',
                boxShadow: `0 0 10px ${dirColor}66`,
              }} />
            </div>
          </>
        ) : (
          /* ── Loading skeleton ── */
          <div style={{ padding: '20px 0' }}>
            <div style={{
              display: 'flex', alignItems: 'center', gap: '12px',
              color: 'var(--text-dim)', fontSize: '1rem',
            }}>
              <span style={{
                display: 'inline-block', width: '10px', height: '10px',
                borderRadius: '50%', background: 'var(--accent)',
                animation: 'pulse 1.4s ease-in-out infinite',
              }} />
              Calculating Cross-Market Conviction…
            </div>
            {[70, 45, 55].map((w, i) => (
              <div key={i} style={{
                height: '10px', borderRadius: '5px',
                background: 'rgba(255,255,255,0.06)',
                marginTop: '14px', width: `${w}%`,
                animation: 'pulse 1.6s ease-in-out infinite',
                animationDelay: `${i * 0.2}s`,
              }} />
            ))}
          </div>
        )}
      </div>

      {/* ── Market pod status grid ──────────────────────────────────────── */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3,1fr)',
        gap: '16px', width: '100%', maxWidth: '620px',
      }}>
        {markets.map(({ key, label, icon }) => {
          const isActive = signal?.market === key
          return (
            <div key={key} style={{
              background: 'var(--panel-bg)',
              border: `1px solid ${isActive ? dirColor + '88' : 'var(--border)'}`,
              borderRadius: '12px', padding: '18px 16px',
              textAlign: 'center',
              boxShadow: isActive ? `0 0 18px ${dirColor}22` : 'none',
              transition: 'all 0.3s ease',
            }}>
              <div style={{ fontSize: '1.4rem', marginBottom: '6px' }}>{icon}</div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: isActive ? dirColor : 'var(--text)' }}>
                {label}
              </div>
              <div style={{
                fontSize: '0.65rem', letterSpacing: '2px',
                color: isActive ? dirColor : 'var(--text-dim)',
                marginTop: '4px', textTransform: 'uppercase',
              }}>
                {isActive ? 'ACTIVE POD' : 'MONITORING'}
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Disclaimer ──────────────────────────────────────────────────── */}
      <div style={{ fontSize: '0.68rem', color: 'var(--text-dim)', textAlign: 'center', maxWidth: '480px' }}>
        Alpha Ranker compares live model probability across all active pods and routes
        execution to the highest statistical edge. All NSE orders are DRY_RUN only.
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
function TradingViewChart({ symbol = 'BTCUSDT' }) {
  let tvSymbol = symbol;
  // NSE indices are blocked from iframe embeds by TradingView, so we must use the continuous future.
  if (symbol === 'BANKNIFTY') tvSymbol = 'NSE:NIFTYBANK';
  else if (symbol === 'BTCUSDT') tvSymbol = 'BINANCE:BTCUSDTPERP';
  else if (symbol === 'EURUSD') tvSymbol = 'FX:EURUSD';

  const src =
    `https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(tvSymbol)}` +
    `&interval=1&theme=dark&style=1&locale=en` +
    `&toolbar_bg=%23141418&enable_publishing=0` +
    `&allow_symbol_change=0&save_image=0&hide_top_toolbar=0` +
    `&withdateranges=1&hide_side_toolbar=0` +
    `&container_id=tradingview_apex` +
    `&timezone=Asia/Kolkata`

  return (
    <section className="panel tradingview-hero" aria-label="Live Price Chart">
      <div className="panel-title-row">
        <div className="panel-title">Live Chart — {tvSymbol}</div>
        <div className="chart-badge">REAL-TIME</div>
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
  const [showCapitalModal, setShowCapitalModal] = useState(false)
  const [cryptoPool, setCryptoPool] = useState('10000')
  const [indiaPool, setIndiaPool] = useState('100000')
  const [forexPool, setForexPool] = useState('10000')
  const [livePrice, setLivePrice] = useState(null)
  const [tickActive, setTickActive] = useState(false)
  const [pollFailures, setPollFailures] = useState(0)

  const getTradingViewSymbol = (tab) => {
    if (tab === 'INDIA') return 'NSE:NIFTYBANK';
    if (tab === 'FOREX') return 'FX:EURUSD';
    return 'BINANCE:BTCUSDTPERP';
  };

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

  const syncCapitalPools = async () => {
    setSavingCapital(true)
    try {
      await fetch(`${API_URL}/api/capital/pools`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          CRYPTO: Number(cryptoPool), 
          INDIA: Number(indiaPool), 
          FOREX: Number(forexPool),
          trade_allocation: Number(tradeAllocationInput)
        }),
      })
      setShowCapitalModal(false)
    } catch (error) {
      console.error('Failed to sync capital pools', error)
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
          <div 
            className="topbar-stat" 
            style={{ cursor: 'pointer' }}
            onClick={() => {
              setCryptoPool(String(session?.capital_pools?.['CRYPTO'] ?? 10000))
              setIndiaPool(String(session?.capital_pools?.['INDIA'] ?? 100000))
              setForexPool(String(session?.capital_pools?.['FOREX'] ?? 10000))
              setShowCapitalModal(true)
            }}
          >
            <span>Capital ({session?.active_tab}) ⚙️</span>
            <strong>
              {session?.active_tab === 'INDIA' 
                ? new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(session?.capital_pools?.['INDIA'] ?? 0)
                : formatCurrency(session?.capital_pools?.[session?.active_tab] ?? session?.total_capital)}
            </strong>
          </div>
          <div className={`topbar-stat pnl ${(session?.sim_pnl ?? 0) >= 0 ? 'pnl--pos' : 'pnl--neg'}`}>
            <span>Session PnL</span>
            <strong>{formatMarketCurrency(session?.sim_pnl, session?.active_tab)}</strong>
          </div>
        </div>
      </header>

      {/* Capital Management Modal */}
      {showCapitalModal && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 10000,
          background: 'rgba(0,0,0,0.8)', display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <div style={{
            background: 'var(--panel-bg)', padding: '24px', borderRadius: '8px', border: '1px solid var(--border)', width: '320px'
          }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '1.1rem' }}>Capital Management</h3>
            
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '4px' }}>CRYPTO Pool (USD)</label>
              <input type="number" value={cryptoPool} onChange={e => setCryptoPool(e.target.value)}
                style={{ width: '100%', padding: '8px', background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '4px' }}
              />
            </div>

            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '4px' }}>INDIA Pool (INR)</label>
              <input type="number" value={indiaPool} onChange={e => setIndiaPool(e.target.value)}
                style={{ width: '100%', padding: '8px', background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '4px' }}
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '4px' }}>FOREX Pool (USD)</label>
              <input type="number" value={forexPool} onChange={e => setForexPool(e.target.value)}
                style={{ width: '100%', padding: '8px', background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '4px' }}
              />
            </div>

            <div style={{ marginBottom: '16px', padding: '12px', background: 'rgba(255,165,0,0.1)', borderRadius: '6px', border: '1px solid rgba(255,165,0,0.2)' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--accent)', fontWeight: 'bold', marginBottom: '4px' }}>Trade Allocation (Override)</label>
              <input type="number" value={tradeAllocationInput} onChange={e => setTradeAllocationInput(e.target.value)}
                style={{ width: '100%', padding: '8px', background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)', borderRadius: '4px' }}
              />
              <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                Set to <strong>0</strong> for dynamic statistical sizing (Kelly Criterion).
              </div>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                onClick={() => setShowCapitalModal(false)}
                style={{ flex: 1, padding: '8px', background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-dim)', borderRadius: '4px', cursor: 'pointer' }}
              >Cancel</button>
              <button 
                onClick={syncCapitalPools} disabled={savingCapital}
                style={{ flex: 1, padding: '8px', background: 'var(--accent)', border: 'none', color: '#fff', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
              >{savingCapital ? 'Saving...' : 'Save Pools'}</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab Navigation ─────────────────────────────────────────────────── */}
      <nav className="tab-navigation" style={{ display: 'flex', gap: '20px', padding: '10px 20px', background: 'var(--panel-bg)', borderBottom: '1px solid var(--border)' }}>
        {['GLOBAL MASTER', 'CRYPTO', 'INDIA', 'FOREX'].map(tab => (
          <button
            key={tab}
            className={`tab-button ${session?.active_tab === tab ? 'active' : ''}`}
            style={{
              background: 'transparent',
              color: session?.active_tab === tab ? 'var(--accent)' : 'var(--text-dim)',
              border: 'none',
              borderBottom: session?.active_tab === tab ? '2px solid var(--accent)' : '2px solid transparent',
              padding: '10px',
              cursor: 'pointer',
              fontWeight: session?.active_tab === tab ? 'bold' : 'normal',
              textTransform: 'uppercase',
            }}
            onClick={async () => {
              try {
                await fetch(`${API_URL}/api/market/tab/${tab}`, { method: 'POST' });
              } catch (e) { console.error('Failed to switch tab', e); }
            }}
          >
            {tab}
          </button>
        ))}
      </nav>

      {session?.active_tab === 'GLOBAL MASTER' ? (
        <GlobalMasterView snapshot={snapshot} />
      ) : (
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

          {/* Trading Style Control */}
          <section className="panel panel--compact">
            <div className="panel-title">Trading Style</div>
            <div className="capital-row" style={{ marginTop: '10px' }}>
              <span className="input-label">Active Profile</span>
              <select 
                className="capital-input" 
                style={{ width: '100%', padding: '8px', background: 'var(--bg)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: '4px' }}
                value={session?.trading_style ?? 'Intraday'}
                onChange={async (e) => {
                  try {
                    await fetch(`${API_URL}/api/session/style/${e.target.value}`, { method: 'POST' });
                  } catch (err) {
                    console.error('Failed to change trading style', err);
                  }
                }}
              >
                <option value="Scalping">Scalping (1m)</option>
                <option value="Intraday">Intraday (15m)</option>
                <option value="Swing">Swing (4h)</option>
              </select>
            </div>
            <div className="metric-list" style={{ marginTop: '10px' }}>
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
                <span className="price-value">{formatMarketCurrency(livePrice ?? market?.close_price, session?.active_tab)}</span>
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
                }}>{formatMarketCurrency(market?.vwap, session?.active_tab)}</strong>
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
          <TradingViewChart symbol={getTradingViewSymbol(session?.active_tab)} />

          {/* ── MID ROW: SmartOrderCard + Active Trades ─────────────────────── */}
          <div className="content-mid">
            <SmartOrderCard
              card={snapshot?.smart_order_card}
              onExecute={executeNow}
              executing={executing}
              activeTiers={activeTiers}
              activeTab={session?.active_tab}
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

              {/* High Conviction Signals */}
              <HighConvictionSignals
                signals={snapshot?.live_signals ?? []}
                activeTab={session?.active_tab}
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
                      <div style={{ fontSize: '0.85rem', fontWeight: '600' }}>
                        {formatMarketCurrency(trade.execution_plan?.est_pnl, session?.active_tab)}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{trade.time}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dimmer)' }}>{trade.trigger_source}</div>
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
      )}
    </div>
  )
}

export default App

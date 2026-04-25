import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = 'ws://127.0.0.1:8080/ws/live'
const API_URL = 'http://127.0.0.1:8080'

const ASSETS = [
  { key: 'BTCUSDT', name: 'Bitcoin (BTC/USDT)', class: 'Crypto' },
  { key: 'NIFTY50', name: 'Nifty 50 Index', class: 'Index' },
  { key: 'EURUSD', name: 'EUR/USD', class: 'Forex' },
]

const STYLES = [
  { key: 'Scalping', desc: 'Ultra-short 1m entries', trail: '0.3%', tp: '0.5%' },
  { key: 'Intraday', desc: 'Same-day, 15m timeframe', trail: '1.0%', tp: '2.0%' },
  { key: 'Swing', desc: 'Multi-day trend following', trail: '3.0%', tp: '8.0%' },
]

function App() {
  const [data, setData] = useState(null)
  const [probHistory, setProbHistory] = useState([])
  const [trades, setTrades] = useState([])
  const [selectedAsset, setSelectedAsset] = useState('BTCUSDT')
  const [selectedStyle, setSelectedStyle] = useState('Intraday')
  const [capital, setCapital] = useState(10000)
  const [capitalInput, setCapitalInput] = useState('10000')
  const [modalTrade, setModalTrade] = useState(null)
  const [showFeatures, setShowFeatures] = useState(false)
  const [wsConnected, setWsConnected] = useState(false)
  const [cycleCount, setCycleCount] = useState(0)
  const [newsList, setNewsList] = useState([])
  const wsRef = useRef(null)
  const canvasRef = useRef(null)

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setWsConnected(true)
        console.log('[WS] Connected to Apex backend')
      }

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data)
          if (payload.type === 'inference') {
            setData(payload)
            setCycleCount(c => c + 1)
            setProbHistory(prev => {
              const next = [...prev, { time: new Date(), prob: payload.probability }]
              return next.slice(-80)
            })
            if (payload.order_id) {
              setTrades(prev => {
                // Strict deduplication by order_id
                if (prev.some(t => t.order_id === payload.order_id)) return prev;
                const next = [{
                  order_id: payload.order_id,
                  time: payload.time || new Date().toLocaleTimeString(),
                  date: payload.date || new Date().toLocaleDateString(),
                  signal: payload.signal,
                  probability: payload.probability,
                  symbol: payload.symbol,
                  entry_price: payload.entry_price,
                  take_profit: payload.take_profit,
                  stop_loss: payload.stop_loss,
                  est_pnl: payload.est_pnl,
                  max_loss: payload.max_loss,
                  position_qty: payload.position_qty,
                  leverage: payload.leverage,
                }, ...prev]
                return next.slice(0, 50)
              })
            }
          }
        } catch (e) {
          console.error('[WS] Parse error:', e)
        }
      }

      ws.onclose = () => {
        setWsConnected(false)
        console.log('[WS] Disconnected, reconnecting in 3s...')
        setTimeout(connect, 3000)
      }

      ws.onerror = () => ws.close()
    }

    connect()
    return () => wsRef.current?.close()
  }, [])

  // Fetch News periodically
  useEffect(() => {
    const fetchNews = async () => {
      try {
        const res = await fetch(`${API_URL}/api/news`)
        const data = await res.json()
        if (data.news) setNewsList(data.news)
      } catch (e) {
        console.error('Failed to fetch news', e)
      }
    }
    fetchNews()
    const interval = setInterval(fetchNews, 60000) // update every minute
    return () => clearInterval(interval)
  }, [])

  // Draw probability chart on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || probHistory.length < 2) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width = canvas.offsetWidth * 2
    const H = canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    const w = W / 2, h = H / 2

    ctx.clearRect(0, 0, w, h)

    // Background
    ctx.fillStyle = 'rgba(10, 10, 15, 0.3)'
    ctx.fillRect(0, 0, w, h)

    // Buy zone
    const buyY = h * (1 - 0.55)
    ctx.fillStyle = 'rgba(63, 185, 80, 0.04)'
    ctx.fillRect(0, 0, w, buyY)

    // Sell zone
    const sellY = h * (1 - 0.45)
    ctx.fillStyle = 'rgba(248, 81, 73, 0.04)'
    ctx.fillRect(0, sellY, w, h - sellY)

    // Grid lines
    ctx.strokeStyle = 'rgba(48, 54, 61, 0.3)'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 10; i++) {
      const y = (h / 10) * i
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
    }

    // Threshold lines
    ctx.setLineDash([4, 4])
    ctx.strokeStyle = 'rgba(63, 185, 80, 0.4)'
    ctx.beginPath(); ctx.moveTo(0, buyY); ctx.lineTo(w, buyY); ctx.stroke()
    ctx.strokeStyle = 'rgba(248, 81, 73, 0.4)'
    ctx.beginPath(); ctx.moveTo(0, sellY); ctx.lineTo(w, sellY); ctx.stroke()
    ctx.strokeStyle = 'rgba(139, 148, 158, 0.2)'
    ctx.beginPath(); ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5); ctx.stroke()
    ctx.setLineDash([])

    // Probability line
    const points = probHistory.map((p, i) => ({
      x: (i / (probHistory.length - 1)) * w,
      y: h * (1 - p.prob),
    }))

    // Fill
    ctx.beginPath()
    ctx.moveTo(points[0].x, h)
    points.forEach(p => ctx.lineTo(p.x, p.y))
    ctx.lineTo(points[points.length - 1].x, h)
    ctx.closePath()
    const grad = ctx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, 'rgba(88, 166, 255, 0.15)')
    grad.addColorStop(1, 'rgba(88, 166, 255, 0)')
    ctx.fillStyle = grad
    ctx.fill()

    // Line
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    points.forEach(p => ctx.lineTo(p.x, p.y))
    ctx.strokeStyle = '#58a6ff'
    ctx.lineWidth = 2
    ctx.stroke()

    // Last point glow
    const last = points[points.length - 1]
    ctx.beginPath()
    ctx.arc(last.x, last.y, 4, 0, Math.PI * 2)
    ctx.fillStyle = '#58a6ff'
    ctx.fill()
    ctx.beginPath()
    ctx.arc(last.x, last.y, 8, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(88, 166, 255, 0.4)'
    ctx.lineWidth = 1
    ctx.stroke()

    // Labels
    ctx.fillStyle = 'rgba(63, 185, 80, 0.6)'
    ctx.font = '10px JetBrains Mono'
    ctx.fillText('BUY', w - 30, buyY - 4)
    ctx.fillStyle = 'rgba(248, 81, 73, 0.6)'
    ctx.fillText('SELL', w - 30, sellY + 12)

  }, [probHistory])

  // Capital sync
  const syncCapital = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/capital`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ total_capital: parseFloat(capitalInput) }),
      })
      const result = await res.json()
      setCapital(result.total_capital)
    } catch (e) {
      console.error('Capital sync failed:', e)
    }
  }, [capitalInput])

  // Fetch trade details
  const openTradeDetail = useCallback(async (orderId) => {
    try {
      const res = await fetch(`${API_URL}/inference/details/${orderId}`)
      const detail = await res.json()
      setModalTrade(detail)
      setShowFeatures(false) // Reset toggle on new open
    } catch (e) {
      console.error('Failed to fetch trade details:', e)
    }
  }, [])

  const prob = data?.probability ?? 0.5
  const signal = data?.signal ?? 'HOLD'
  const pnl = data?.sim_pnl ?? 0
  const sentiment = data?.sentiment ?? 0.5

  const probClass = prob > 0.55 ? 'text-bullish' : prob < 0.45 ? 'text-bearish' : 'text-neutral'
  const signalClass = signal === 'BUY' ? 'text-bullish' : signal === 'SELL' ? 'text-bearish' : 'text-neutral'
  const pnlClass = pnl > 0 ? 'text-bullish' : pnl < 0 ? 'text-bearish' : 'text-neutral'
  const sentClass = sentiment > 0.6 ? 'text-bullish' : sentiment < 0.4 ? 'text-bearish' : 'text-neutral'

  const styleInfo = STYLES.find(s => s.key === selectedStyle)

  return (
    <>
      {/* ── Top Bar ── */}
      <div className="topbar">
        <div className="topbar-logo">⚡ APEX COMMAND CENTER</div>
        <div className="topbar-controls">
          <select className="topbar-select" value={selectedAsset} onChange={e => setSelectedAsset(e.target.value)}>
            {ASSETS.map(a => <option key={a.key} value={a.key}>{a.name}</option>)}
          </select>
          <select className="topbar-select" value={selectedStyle} onChange={e => setSelectedStyle(e.target.value)}>
            {STYLES.map(s => <option key={s.key} value={s.key}>{s.key} — {s.desc}</option>)}
          </select>
          <div className="topbar-status">
            <span className={`status-dot ${wsConnected ? 'live' : ''}`}></span>
            <span>{wsConnected ? 'LIVE' : 'CONNECTING...'}</span>
            <span style={{ marginLeft: 8, color: '#6e7681' }}>Cycle #{cycleCount}</span>
          </div>
        </div>
      </div>

      <div className="broker-layout">
        <aside className="broker-sidebar">
          {/* ── Capital Control ── */}
          <div className="capital-section glass-card">
            <div className="capital-input-group">
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Balance:</span>
              <input
                className="capital-input"
                type="number"
                value={capitalInput}
                onChange={e => setCapitalInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && syncCapital()}
              />
              <button className="capital-btn" onClick={syncCapital}>Sync</button>
            </div>
            <div className="capital-metrics">
              <span>Trail: {styleInfo?.trail}</span>
              <span>TP: {styleInfo?.tp}</span>
              <span>VWAP: <strong style={{ color: 'var(--purple)' }}>{data?.vwap?.toFixed(1) ?? '—'}</strong></span>
              <span>ATR: <strong style={{ color: 'var(--pink)' }}>{data?.atr?.toFixed(2) ?? '—'}</strong></span>
            </div>
          </div>

          {/* ── HUD Widgets ── */}
          <div className="sidebar-huds">
            <div className="glass-card hud-card">
              <div className="hud-label">Bullish Probability</div>
              <div className={`hud-value ${probClass}`}>{prob.toFixed(4)}</div>
              <div className="signal-bar">
                <div className="signal-marker" style={{ left: `${prob * 100}%` }}></div>
              </div>
            </div>

            <div className="glass-card hud-card">
              <div className="hud-label">Active Signal</div>
              <div className={`hud-value ${signalClass}`}>{signal}</div>
              <div className="hud-sub">BUY &gt;0.55 · SELL &lt;0.45</div>
            </div>

            <div className="glass-card hud-card">
              <div className="hud-label">Session PnL</div>
              <div className={`hud-value ${pnlClass}`}>${pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}</div>
            </div>

            <div className="glass-card hud-card">
              <div className="hud-label">Macro Sentiment</div>
              <div className={`hud-value ${sentClass}`}>{sentiment.toFixed(2)}</div>
            </div>
          </div>

          {/* ── Probability Chart ── */}
          <div className="glass-card sidebar-chart">
            <div className="chart-title">📈 Oscillator</div>
            <div className="chart-canvas" style={{ height: '140px' }}>
              <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }}></canvas>
            </div>
          </div>

          {/* ── News Widget ── */}
          <div className="glass-card news-widget">
            <div className="chart-title">📰 Global Market News</div>
            <div className="news-list">
              {newsList.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', padding: '10px' }}>Loading headlines...</div>
              ) : (
                newsList.map((n, i) => (
                  <div key={i} className="news-item">• {n}</div>
                ))
              )}
            </div>
          </div>
        </aside>

        <main className="broker-main">
          <div className="feed-header">
            <h2>Active Operations</h2>
            <div className="feed-status">{wsConnected ? 'LIVE FEED ACTIVE' : 'RECONNECTING...'}</div>
          </div>
          
          <div className="active-trade-panel">
            {trades.length === 0 ? (
              <div className="empty-feed">
                <div className="radar-spinner"></div>
                Waiting for signal conviction...
              </div>
            ) : (
              <div className="active-trade-card glass-card">
                <div className="active-card-header">
                  <span className={`active-badge ${trades[0].signal.toLowerCase()}`}>{trades[0].signal}</span>
                  <h3>{trades[0].symbol}</h3>
                  <span className="active-time">{trades[0].time}</span>
                </div>
                
                <div className="active-execution-details">
                  <div className="exec-stat">
                    <span className="stat-label">Entry</span>
                    <span className="stat-value">${trades[0].entry_price?.toFixed(2)}</span>
                  </div>
                  <div className="exec-stat">
                    <span className="stat-label">Target</span>
                    <span className="stat-value text-bullish">${trades[0].take_profit?.toFixed(2)}</span>
                  </div>
                  <div className="exec-stat">
                    <span className="stat-label">Stop</span>
                    <span className="stat-value text-bearish">${trades[0].stop_loss?.toFixed(2)}</span>
                  </div>
                  <div className="exec-stat">
                    <span className="stat-label">Size</span>
                    <span className="stat-value">{trades[0].position_qty?.toFixed(4)} ({trades[0].leverage}x)</span>
                  </div>
                </div>

                <div className="active-card-footer">
                  <span style={{color: 'var(--text-muted)'}}>#{trades[0].order_id}</span>
                  <button className="view-details-btn" onClick={() => openTradeDetail(trades[0].order_id)}>
                    View Full Ticket ➔
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="feed-header" style={{borderTop: '1px solid var(--border)', marginTop: 'auto', background: 'transparent'}}>
            <h2 style={{fontSize: '0.9rem'}}>Execution History</h2>
          </div>
          
          <div className="history-feed">
            {trades.slice(1).map((t, i) => (
              <div
                key={i}
                className={`history-item ${t.signal.toLowerCase()}`}
                onClick={() => openTradeDetail(t.order_id)}
              >
                <span className={`history-badge ${t.signal.toLowerCase()}`}>{t.signal}</span>
                <span className="history-symbol">{t.symbol}</span>
                <span className="history-price">Entry: ${t.entry_price?.toFixed(2)}</span>
                <span className="history-time">{t.time}</span>
              </div>
            ))}
            {trades.length <= 1 && (
              <div className="empty-history">No past executions in this session.</div>
            )}
          </div>
        </main>
      </div>

      {/* ── Trade Detail Modal ── */}
      {modalTrade && (
        <div className="modal-overlay" onClick={() => setModalTrade(null)}>
          <div className="modal-card broker-ticket-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-title">
              <div className="ticket-header-info">
                <span className={`ticket-badge ${modalTrade.signal.toLowerCase()}`}>
                  {modalTrade.signal}
                </span>
                <span style={{ marginLeft: 12 }}>{modalTrade.symbol}</span>
                <span className="ticket-order-id">#{modalTrade.order_id}</span>
              </div>
              <span className="modal-close" onClick={() => setModalTrade(null)}>✕</span>
            </div>
            
            <div className="ticket-meta">
              <span>{modalTrade.date} · {modalTrade.time}</span>
              <span>Style: <strong style={{color: 'var(--accent)'}}>{modalTrade.style}</strong></span>
              <span>Timeframe: <strong>{modalTrade.timeframe}</strong></span>
            </div>

            <div className="ticket-execution-grid">
              <div className="exec-item">
                <div className="exec-label">Entry Price</div>
                <div className="exec-value">${modalTrade.entry_price?.toFixed(2) ?? '—'}</div>
              </div>
              <div className="exec-item">
                <div className="exec-label">Take Profit (+{modalTrade.tp_pct}%)</div>
                <div className="exec-value text-bullish">${modalTrade.take_profit?.toFixed(2) ?? '—'}</div>
              </div>
              <div className="exec-item">
                <div className="exec-label">Stop Loss (-{modalTrade.sl_pct}%)</div>
                <div className="exec-value text-bearish">${modalTrade.stop_loss?.toFixed(2) ?? '—'}</div>
              </div>
              
              <div className="exec-item">
                <div className="exec-label">Position Qty</div>
                <div className="exec-value">{modalTrade.position_qty?.toFixed(4) ?? '—'}</div>
              </div>
              <div className="exec-item">
                <div className="exec-label">Est. PnL</div>
                <div className="exec-value text-bullish">+${modalTrade.est_pnl?.toFixed(2) ?? '—'}</div>
              </div>
              <div className="exec-item">
                <div className="exec-label">Max Risk</div>
                <div className="exec-value text-bearish">-${modalTrade.max_loss?.toFixed(2) ?? '—'}</div>
              </div>
            </div>

            <div className="ticket-metrics-row">
              <div className="metric-pill">
                <span className="pill-label">Risk/Reward:</span>
                <span className="pill-val">1:{modalTrade.risk_reward?.toFixed(2) ?? '—'}</span>
              </div>
              <div className="metric-pill">
                <span className="pill-label">Allocation:</span>
                <span className="pill-val">${modalTrade.allocation?.toFixed(2) ?? '—'}</span>
              </div>
              <div className="metric-pill">
                <span className="pill-label">AI Confidence:</span>
                <span className={`pill-val ${modalTrade.probability > 0.55 ? 'text-bullish' : 'text-bearish'}`}>
                  {(modalTrade.probability * 100)?.toFixed(1)}%
                </span>
              </div>
            </div>

            {/* AI Diagnostics Toggle */}
            <div 
              className="ai-toggle-btn"
              onClick={() => setShowFeatures(!showFeatures)}
            >
              {showFeatures ? '▼ Hide AI Feature Diagnostics' : '▶ View AI Feature Diagnostics'}
            </div>

            {showFeatures && (
              <div className="feature-grid ticket-features">
                {modalTrade.features && Object.entries(modalTrade.features).map(([k, v]) => (
                  <div className="feature-item" key={k}>
                    <span className="feature-name">{k}</span>
                    <span className="feature-val">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                  </div>
                ))}
              </div>
            )}
            
            <button className="confirm-trade-btn" onClick={() => setModalTrade(null)}>
              Acknowledge execution ticket
            </button>
          </div>
        </div>
      )}
    </>
  )
}

export default App

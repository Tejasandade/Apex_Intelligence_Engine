function formatCurrency(value, activeTab) {
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

function getConfidenceTier(probability) {
  if (probability > 0.80 || probability < 0.20) return 3
  if (probability > 0.65 || probability < 0.35) return 2
  if (probability > 0.55 || probability < 0.45) return 1
  return 0
}

// ── Regime badge helpers ────────────────────────────────────────────────────
function RegimeBadge({ regime }) {
  const isTrending = regime === 'Trending'
  const color  = isTrending ? '#22d3ee' : '#f59e0b'   // cyan vs amber
  const bg     = isTrending ? 'rgba(34,211,238,0.10)' : 'rgba(245,158,11,0.10)'
  const border = isTrending ? 'rgba(34,211,238,0.35)' : 'rgba(245,158,11,0.35)'
  const icon   = isTrending ? '📈' : '🌀'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '5px',
      padding: '3px 10px', borderRadius: '20px',
      background: bg, border: `1px solid ${border}`,
      color, fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.5px',
      boxShadow: `0 0 8px ${color}33`,
    }}>
      {icon} {regime ?? 'Unknown'}
    </span>
  )
}

function SmartOrderCard({ card, onExecute, executing, activeTiers = [], activeTab = 'CRYPTO' }) {
  const plan      = card?.execution_plan
  const analysis  = card?.signal_analysis
  const bullish   = card?.bullish_probability ?? 0.5
  const signalClass = card?.signal?.toLowerCase?.() ?? 'hold'

  const features  = card?.features || {}
  const sentiment = card?.sentiment ?? 0.5

  // ── Council Logic ────────────────────────────────────────────────────────
  let quantGreen = false, structureGreen = false, sentimentGreen = false
  quantGreen = (bullish > 0.60 || bullish < 0.40)
  if (bullish > 0.60) {
    structureGreen = (
      features.fvg_signal > 0 ||
      features.structure_break_signal > 0 ||
      features.liquidity_sweep_signal > 0 ||
      features.cvd_cumulative > 0
    )
    sentimentGreen = sentiment > 0.52
  } else if (bullish < 0.40) {
    structureGreen = (
      features.fvg_signal < 0 ||
      features.structure_break_signal < 0 ||
      features.liquidity_sweep_signal < 0 ||
      features.cvd_cumulative < 0
    )
    sentimentGreen = sentiment < 0.48
  } else {
    structureGreen = features.fvg_signal !== 0 || features.structure_break_signal !== 0
    sentimentGreen = Math.abs(sentiment - 0.5) > 0.03
  }

  const consensusReached  = quantGreen && structureGreen && sentimentGreen
  const currentTier       = getConfidenceTier(bullish)
  const isExecutionBlocked = activeTiers.length >= 3 || (currentTier > 0 && activeTiers.includes(currentTier))

  // ── Regime data ──────────────────────────────────────────────────────────
  const regime    = analysis?.regime_classification ?? 'Unknown'
  const adxVal    = analysis?.adx_value ?? 0
  const chopVal   = analysis?.chop_value ?? 61.8
  const atrDist   = analysis?.atr_distance ?? 0

  return (
    <section className="smart-card">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="smart-card__header">
        <div>
          <div className="eyebrow">Signal Center</div>
          <h1>Smart Order Card</h1>
        </div>
        <div className={`signal-pill signal-pill--${signalClass}`}>
          {card?.signal ?? 'HOLD'}
        </div>
      </div>

      {/* ── Topline meta ────────────────────────────────────────────────── */}
      <div className="smart-card__topline">
        <div>
          <div className="meta-label">Provider</div>
          <div className="meta-value">{card?.provider ?? 'binance_futures'}</div>
        </div>
        <div>
          <div className="meta-label">Style ({plan?.timeframe ?? '15m'})</div>
          <div className="meta-value" style={{ textTransform: 'uppercase' }}>{plan?.style ?? 'INTRADAY'}</div>
        </div>
        <div>
          <div className="meta-label">Status</div>
          <div className="meta-value">{card?.status ?? 'monitoring'}</div>
        </div>
      </div>

      {card?.warmup_message ? (
        <div className="smart-card__warmup">{card.warmup_message}</div>
      ) : null}

      {/* ── Market Regime Panel (Epic 27) ────────────────────────────────── */}
      <article style={{
        marginTop: '12px',
        padding: '12px 14px',
        borderRadius: '10px',
        background: regime === 'Trending'
          ? 'rgba(34,211,238,0.05)'
          : 'rgba(245,158,11,0.05)',
        border: regime === 'Trending'
          ? '1px solid rgba(34,211,238,0.18)'
          : '1px solid rgba(245,158,11,0.18)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '10px',
      }}>
        {/* Left: badge + label */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div>
            <div style={{ fontSize: '0.62rem', letterSpacing: '2px', color: 'var(--text-dim)', marginBottom: '5px', textTransform: 'uppercase' }}>
              Market Regime
            </div>
            <RegimeBadge regime={regime} />
          </div>
        </div>

        {/* Right: ADX / CHOP / ATR Distance mini-stats */}
        <div style={{ display: 'flex', gap: '16px' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)', letterSpacing: '1px', textTransform: 'uppercase' }}>ADX</div>
            <div style={{
              fontSize: '0.9rem', fontWeight: 700,
              color: adxVal > 25 ? '#22d3ee' : 'var(--text-dim)',
            }}>{adxVal > 0 ? adxVal.toFixed(1) : '--'}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)', letterSpacing: '1px', textTransform: 'uppercase' }}>CHOP</div>
            <div style={{
              fontSize: '0.9rem', fontWeight: 700,
              color: chopVal < 38.2 ? '#22d3ee' : chopVal > 61.8 ? '#f59e0b' : 'var(--text)',
            }}>{chopVal > 0 ? chopVal.toFixed(1) : '--'}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '0.6rem', color: 'var(--text-dim)', letterSpacing: '1px', textTransform: 'uppercase' }}>ATR%</div>
            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text)' }}>
              {atrDist > 0 ? `${atrDist.toFixed(2)}%` : '--'}
            </div>
          </div>
        </div>
      </article>

      {/* ── Council Logic ───────────────────────────────────────────────── */}
      <div className="council-grid">
        <div className="eyebrow" style={{ gridColumn: '1 / -1', marginBottom: '8px' }}>Council Logic (3-Agent Vote)</div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div className="council-bulb">
            <span className={`dot ${quantGreen ? 'dot-green' : 'dot-gray'}`} />
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Quant</span>
          </div>
          <div className="council-bulb">
            <span className={`dot ${structureGreen ? 'dot-green' : 'dot-gray'}`} />
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Structure</span>
          </div>
          <div className="council-bulb">
            <span className={`dot ${sentimentGreen ? 'dot-green' : 'dot-gray'}`} />
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Sentiment</span>
          </div>
          {consensusReached && (
            <div className="chart-badge" style={{ marginLeft: 'auto', background: 'var(--bullish-soft)', borderColor: 'var(--bullish)', color: 'var(--bullish)' }}>
              COUNCIL CONSENSUS
            </div>
          )}
        </div>
      </div>

      {/* ── Technical + Sentiment analysis ─────────────────────────────── */}
      <div className="analysis-grid">
        <article className="analysis-card">
          <div className="eyebrow">Technical Confluence</div>
          <div className="analysis-score" style={{ color: quantGreen ? 'var(--bullish)' : 'var(--text-dim)' }}>
            {(analysis?.technical_score * 100 || 0).toFixed(0)}% CONFIDENCE
          </div>
          <p style={{ fontSize: '0.75rem', lineHeight: '1.4', color: 'var(--text-dim)' }}>
            {analysis?.technical_summary ?? 'Model is analyzing technical clusters and price action divergence.'}
          </p>
        </article>
        <article className="analysis-card">
          <div className="eyebrow">Macro &amp; Sentiment</div>
          <div className="analysis-score" style={{ color: sentimentGreen ? 'var(--bullish)' : 'var(--text-dim)' }}>
            {(analysis?.sentiment_score * 100 || 0).toFixed(0)}% SENTIMENT
          </div>
          <p style={{ fontSize: '0.75rem', lineHeight: '1.4', color: 'var(--text-dim)' }}>
            {analysis?.sentiment_summary ?? 'Sentiment wire scanning headlines for institutional positioning.'}
          </p>
        </article>
      </div>

      {/* ── Institutional Flow ──────────────────────────────────────────── */}
      <article className="analysis-card" style={{ marginTop: '12px', background: 'rgba(0,188,212,0.05)', border: '1px solid rgba(0,188,212,0.1)' }}>
        <div className="eyebrow" style={{ color: 'var(--accent)' }}>Institutional Flow Analysis</div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text)' }}>
            {card?.structural_edge || 'Scanning for Liquidity Sweeps & FVG Gaps...'}
          </div>
          <div className={`dot ${structureGreen ? 'dot-green' : 'dot-gray'}`} style={{ width: '8px', height: '8px' }} />
        </div>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '4px', letterSpacing: '0.5px' }}>
          CVD: {features.cvd_cumulative?.toFixed(2) ?? '0.00'} | FVG: {features.fvg_signal > 0 ? 'BULLISH GAP' : features.fvg_signal < 0 ? 'BEARISH GAP' : 'NEUTRAL'}
        </div>
      </article>

      {/* ── Structural Edge ─────────────────────────────────────────────── */}
      <article className={`structure-card ${card?.structural_confluence ? 'structure-card--confluence' : ''}`}>
        <div className="eyebrow">Structural Edge</div>
        <div className="structure-card__headline">{card?.structural_edge ?? 'No major institutional structure edge detected.'}</div>
        <div className="structure-card__score">
          Structure Score: {((card?.structural_score ?? 0) * 100).toFixed(0)} / 100
        </div>
        <div className="structure-chip-row">
          {(card?.structure_signals?.length ? card.structure_signals : ['Monitoring structure']).map((signal) => (
            <span key={signal} className="structure-chip">{signal}</span>
          ))}
        </div>
      </article>

      {/* ── Execution stats ─────────────────────────────────────────────── */}
      <div className="execution-grid">
        <div className="execution-stat">
          <span>Entry</span>
          <strong>{formatCurrency(plan?.entry_price, activeTab)}</strong>
        </div>
        <div className="execution-stat">
          <span>Target</span>
          <strong>{formatCurrency(plan?.take_profit, activeTab)}</strong>
        </div>
        <div className="execution-stat">
          <span>Stop</span>
          <strong>{formatCurrency(plan?.stop_loss, activeTab)}</strong>
        </div>
        <div className="execution-stat">
          <span>R/R Ratio</span>
          <strong style={{ color: plan?.risk_reward > 2 ? 'var(--bullish)' : 'var(--text)' }}>{plan?.risk_reward?.toFixed(2) ?? '0.00'}</strong>
        </div>
        <div className="execution-stat">
          <span>Position PnL</span>
          <strong className={plan?.est_pnl >= 0 ? 'text-positive' : 'text-negative'}>
            {formatCurrency(plan?.est_pnl, activeTab)}
          </strong>
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────────── */}
      <div className="smart-card__footer">
        <div className="override-copy">
          {isExecutionBlocked
            ? '⛔ Execute locked — tier slot occupied or max trades (3) reached.'
            : card?.requires_manual_approval
            ? 'Manual approval is active. Use Execute Now to override and stage the order immediately.'
            : 'Execution slot available. Execute Now remains available as a manual command center override.'}
        </div>
        {isExecutionBlocked && (
          <div className="active-trade-badge">⛔ Slot locked</div>
        )}
        <button
          type="button"
          className="execute-button"
          disabled={!plan || executing || isExecutionBlocked}
          onClick={onExecute}
        >
          {executing ? 'Routing...' : isExecutionBlocked ? 'Position Active' : card?.execute_label ?? 'Execute Now'}
        </button>
      </div>
    </section>
  )
}

export default SmartOrderCard

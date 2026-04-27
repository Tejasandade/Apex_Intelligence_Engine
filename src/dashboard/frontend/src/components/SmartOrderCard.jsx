function formatCurrency(value, activeTab) {
  if (value == null || Number.isNaN(Number(value))) {
    return '--'
  }

  if (activeTab === 'INDIA') {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2,
    }).format(Number(value))
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(Number(value))
}

function getConfidenceTier(probability) {
  if (probability > 0.80 || probability < 0.20) return 3;
  if (probability > 0.65 || probability < 0.35) return 2;
  if (probability > 0.55 || probability < 0.45) return 1;
  return 0;
}

function SmartOrderCard({ card, onExecute, executing, activeTiers = [], activeTab = 'CRYPTO' }) {
  const plan = card?.execution_plan
  const bullish = card?.bullish_probability ?? 0.5
  const signalClass = card?.signal?.toLowerCase?.() ?? 'hold'

  const features = card?.features || {}
  const sentiment = card?.sentiment ?? 0.5

  let quantGreen = false, structureGreen = false, sentimentGreen = false

  // Quant: model has strong directional conviction
  quantGreen = (bullish > 0.60 || bullish < 0.40)

  if (bullish > 0.60) {
    // Bullish scenario — check structural and sentiment alignment
    structureGreen = (
      features.fvg_signal > 0 ||
      features.structure_break_signal > 0 ||
      features.liquidity_sweep_signal > 0 ||
      features.cvd_cumulative > 0
    )
    sentimentGreen = sentiment > 0.52
  } else if (bullish < 0.40) {
    // Bearish scenario — check structural and sentiment alignment
    structureGreen = (
      features.fvg_signal < 0 ||
      features.structure_break_signal < 0 ||
      features.liquidity_sweep_signal < 0 ||
      features.cvd_cumulative < 0
    )
    sentimentGreen = sentiment < 0.48
  } else {
    // Neutral zone — use any structural signal or near-neutral sentiment
    structureGreen = features.fvg_signal !== 0 || features.structure_break_signal !== 0
    sentimentGreen = Math.abs(sentiment - 0.5) > 0.03
  }

  const consensusReached = quantGreen && structureGreen && sentimentGreen
  const currentTier = getConfidenceTier(bullish)
  const isExecutionBlocked = activeTiers.length >= 3 || (currentTier > 0 && activeTiers.includes(currentTier))

  return (
    <section className="smart-card">
      <div className="smart-card__header">
        <div>
          <div className="eyebrow">Signal Center</div>
          <h1>Smart Order Card</h1>
        </div>
        <div className={`signal-pill signal-pill--${signalClass}`}>
          {card?.signal ?? 'HOLD'}
        </div>
      </div>

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

      <div className="council-grid">
        <div className="eyebrow" style={{ gridColumn: '1 / -1', marginBottom: '8px' }}>Council Logic (3-Agent Vote)</div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div className="council-bulb">
            <span className={`dot ${quantGreen ? 'dot-green' : 'dot-gray'}`}></span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Quant</span>
          </div>
          <div className="council-bulb">
            <span className={`dot ${structureGreen ? 'dot-green' : 'dot-gray'}`}></span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Structure</span>
          </div>
          <div className="council-bulb">
            <span className={`dot ${sentimentGreen ? 'dot-green' : 'dot-gray'}`}></span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Sentiment</span>
          </div>
          {consensusReached && (
            <div className="chart-badge" style={{ marginLeft: 'auto', background: 'var(--bullish-soft)', borderColor: 'var(--bullish)', color: 'var(--bullish)' }}>
              COUNCIL CONSENSUS
            </div>
          )}
        </div>
      </div>

      <div className="analysis-grid">
        <article className="analysis-card">
          <div className="eyebrow">Technical Analysis</div>
          <div className="analysis-score">
            {(card?.signal_analysis?.technical_score * 100 || 0).toFixed(0)} / 100
          </div>
          <p>{card?.signal_analysis?.technical_summary ?? 'Technical context is loading.'}</p>
        </article>
        <article className="analysis-card">
          <div className="eyebrow">Sentiment Analysis</div>
          <div className="analysis-score">
            {(card?.signal_analysis?.sentiment_score * 100 || 0).toFixed(0)} / 100
          </div>
          <p>{card?.signal_analysis?.sentiment_summary ?? 'Sentiment context is loading.'}</p>
        </article>
      </div>

      <article className={`structure-card ${card?.structural_confluence ? 'structure-card--confluence' : ''}`}>
        <div className="eyebrow">Structural Edge</div>
        <div className="structure-card__headline">{card?.structural_edge ?? 'No major institutional structure edge detected.'}</div>
        <div className="structure-card__score">
          Structure Score: {((card?.structural_score ?? 0) * 100).toFixed(0)} / 100
        </div>
        <div className="structure-chip-row">
          {(card?.structure_signals?.length ? card.structure_signals : ['Monitoring structure']).map((signal) => (
            <span key={signal} className="structure-chip">
              {signal}
            </span>
          ))}
        </div>
      </article>

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
          <span>R/R</span>
          <strong>{plan?.risk_reward?.toFixed(2) ?? '--'}</strong>
        </div>
      </div>

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

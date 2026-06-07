/**
 * Apex Intelligence Engine V5 — Dashboard Client
 * =================================================
 * Connects to the live runner's WebSocket API and updates the UI
 * in real-time with signals, council decisions, trades, and P&L.
 *
 * Also supports simulation mode for demo/testing.
 */

// ── State ────────────────────────────────────────────────────────────────────
const state = {
    connected: false,
    mode: 'PAPER',
    startTime: Date.now(),
    balance: 10000,
    initialBalance: 10000,
    totalPnl: 0,
    totalTrades: 0,
    winningTrades: 0,
    drawdown: 0,
    regime: '—',
    adx: 0,
    signals: 0,
    candles: 0,
    position: null,
    lastEnsemble: null,
    lastCouncil: null,
};

// ── DOM Elements ─────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── WebSocket Connection ─────────────────────────────────────────────────────
let ws = null;
const urlParams = new URLSearchParams(window.location.search);
const wsPort = urlParams.get('port') || '8765';
const WS_URL = `ws://localhost:${wsPort}`;

function connectWebSocket() {
    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            state.connected = true;
            updateConnectionStatus(true);
            console.log('[Apex] Connected to live engine');
            
            // Clear old session data from DOM to prevent ghosts after engine restart
            if ($('tradeHistory')) $('tradeHistory').innerHTML = '<div class="trade-empty">No trades yet</div>';
            if ($('signalFeed')) $('signalFeed').innerHTML = '<div class="signal-empty">Waiting for signals...</div>';
            
            // Reset state
            state.totalPnl = 0;
            state.totalTrades = 0;
            state.winningTrades = 0;
            state.balance = state.initialBalance;
            refreshMetrics();
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            } catch (e) {
                console.warn('[Apex] Parse error:', e);
            }
        };

        ws.onclose = () => {
            state.connected = false;
            updateConnectionStatus(false);
            // Reconnect after 3s
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = () => {
            // Will trigger onclose → reconnect
        };
    } catch (e) {
        // Server not running — use simulation
        console.log('[Apex] Server not available, running in simulation mode');
        startSimulation();
    }
}

function handleMessage(msg) {
    switch (msg.type) {
        case 'status':
            updateStatus(msg.data);
            break;
        case 'signal':
            addSignal(msg.data);
            break;
        case 'council':
            updateCouncil(msg.data);
            break;
        case 'ensemble':
            updateEnsemble(msg.data);
            break;
        case 'trade_open':
            updatePosition(msg.data);
            break;
        case 'trade_close':
            closeTrade(msg.data);
            break;
        case 'tick':
            updateTick(msg.data);
            break;
        case 'trail_update':
            // Add trail update handling (could log it, or update position)
            break;
    }
}

// ── UI Update Functions ──────────────────────────────────────────────────────

function updateConnectionStatus(connected) {
    const el = $('wsStatus');
    const text = $('wsStatusText');
    if (connected) {
        el.classList.add('connected');
        text.textContent = 'Connected';
    } else {
        el.classList.remove('connected');
        text.textContent = 'Disconnected';
    }
}

function updateStatus(data) {
    state.balance = data.balance || state.balance;
    state.totalPnl = data.total_pnl || 0;
    state.totalTrades = data.total_trades || 0;
    state.winningTrades = data.winning_trades || 0;
    state.drawdown = data.drawdown_pct || 0;
    state.regime = data.regime || '—';
    state.adx = data.adx || 0;
    state.signals = data.signals || 0;
    state.candles = data.candles || 0;
    state.currency = data.currency || '$';
    
    if (data.initial_balance) {
        state.initialBalance = data.initial_balance;
    }
    
    // Cooldown status
    if (data.cooldown_remaining && data.cooldown_remaining > 0) {
        state.cooldown = data.cooldown_remaining;
    } else {
        state.cooldown = 0;
    }

    refreshMetrics();
}

function refreshMetrics() {
    const delta = state.balance - state.initialBalance;
    const deltaPct = (delta / state.initialBalance) * 100;

    $('balance').textContent = formatCurrency(state.balance);
    $('balanceDelta').textContent = formatCurrency(delta, true);
    $('balanceDelta').className = `metric-card__delta ${delta >= 0 ? 'positive' : 'negative'}`;

    $('totalPnl').textContent = formatCurrency(state.totalPnl, true);
    $('totalPnl').style.color = state.totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
    $('totalPnlPct').textContent = `${deltaPct >= 0 ? '+' : ''}${deltaPct.toFixed(2)}%`;
    $('totalPnlPct').className = `metric-card__delta ${deltaPct >= 0 ? 'positive' : 'negative'}`;

    const winRate = state.totalTrades > 0
        ? ((state.winningTrades / state.totalTrades) * 100).toFixed(0) + '%'
        : '—';
    $('winRate').textContent = winRate;
    $('totalTrades').textContent = `${state.totalTrades} trades`;

    $('drawdown').textContent = state.drawdown.toFixed(1) + '%';
    $('drawdown').style.color = state.drawdown > 5 ? 'var(--red)' : 'var(--text-primary)';

    $('regime').textContent = state.regime;
    $('regimeADX').textContent = `ADX: ${state.adx.toFixed(1)}`;

    $('signalCount').textContent = state.signals;
    $('candleCount').textContent = `${state.candles} candles`;
}

function addSignal(data) {
    state.signals++;
    const feed = $('signalFeed');

    // Remove empty placeholder
    const empty = feed.querySelector('.signal-empty');
    if (empty) empty.remove();

    const blocked = data.blocked || false;
    const dir = data.direction || 'SELL';
    const cls = blocked ? 'blocked' : dir.toLowerCase();
    const arrow = dir === 'BUY' ? '▲' : '▼';
    const icon = blocked ? '🚫' : (dir === 'BUY' ? '🟢' : '🔴');

    const priceStr = `${state.currency || '$'}${(data.price || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    const convStr = `${((data.conviction || 0) * 100).toFixed(1)}%`;
    const councilStr = `${((data.council_score || 0) * 100).toFixed(0)}%`;

    const card = document.createElement('div');
    card.className = `signal-card signal-card--${cls}`;
    
    // Format reason text to look like a terminal log if it's long
    let reasonHtml = '';
    if (data.reason) {
        if (data.reason.length > 100) {
            let formattedReason = data.reason;
            if (formattedReason.includes('[Vote.')) {
                formattedReason = formattedReason
                    .replace(/\[Vote\.APPROVE\]/g, '<span class="log-badge log-badge--approve">APPROVE</span>')
                    .replace(/\[Vote\.REJECT\]/g, '<span class="log-badge log-badge--reject">REJECT</span>')
                    .replace(/\[Vote\.ABSTAIN\]/g, '<span class="log-badge log-badge--abstain">ABSTAIN</span>')
                    .replace(/❌/g, '<span style="color:var(--red);">❌</span>')
                    .replace(/✅/g, '<span style="color:var(--green);">✅</span>')
                    .replace(/⚪/g, '<span style="color:var(--blue);">⚪</span>')
                    .replace(/ \| /g, '<br><span style="color:var(--text-muted); opacity: 0.5; margin-right: 6px;">|</span>');
            }
            reasonHtml = `<div class="signal-card__reason signal-card__reason--log">${formattedReason}</div>`;
        } else {
            reasonHtml = `<div class="signal-card__reason">${data.reason}</div>`;
        }
    }

    card.innerHTML = `
        <div class="signal-card__header">
            <div class="signal-card__badge-group">
                <span class="signal-badge signal-badge--${cls}">
                    ${icon} ${dir} ${(data.symbol || 'BTCUSDT').toUpperCase()}
                </span>
                ${blocked ? '<span class="signal-badge signal-badge--blocked-tag">VETOED</span>' : '<span class="signal-badge signal-badge--approved-tag">APPROVED</span>'}
            </div>
            <span class="signal-card__time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="signal-card__metrics">
            <div class="signal-metric">
                <span class="signal-metric__label">PRICE</span>
                <span class="signal-metric__value">${priceStr}</span>
            </div>
            <div class="signal-metric">
                <span class="signal-metric__label">CONVICTION</span>
                <span class="signal-metric__value">${convStr}</span>
            </div>
            <div class="signal-metric">
                <span class="signal-metric__label">COUNCIL</span>
                <span class="signal-metric__value">${councilStr}</span>
            </div>
        </div>
        ${reasonHtml}
    `;

    feed.insertBefore(card, feed.firstChild);

    // Keep max 20 signals
    while (feed.children.length > 20) {
        feed.removeChild(feed.lastChild);
    }

    refreshMetrics();
}

function updateCouncil(data) {
    state.lastCouncil = data;
    const votes = data.votes || {};
    const grid = document.querySelector('.advisor-grid');
    if (grid) {
        // Build dynamic HTML for advisors based on payload
        let html = '';
        for (const [name, voteObj] of Object.entries(votes)) {
            const v = voteObj.vote || 'ABSTAIN';
            const iconStr = v === 'APPROVE' ? '✅' : v === 'REJECT' ? '❌' : '⚪';
            const vClass = `advisor__vote--${v.toLowerCase()}`;
            const w = voteObj.weight ? voteObj.weight.toFixed(1) : '1.0';
            
            // Map simple icon (fallback to abstract)
            let icon = '🏛️';
            if (name.includes('Momentum')) icon = '📈';
            else if (name.includes('Structure')) icon = '🏗️';
            else if (name.includes('Volume')) icon = '📊';
            else if (name.includes('Regime')) icon = '🌊';
            else if (name.includes('Risk')) icon = '🛡️';
            else if (name.includes('Session')) icon = '⏱️';
            else if (name.includes('VIX')) icon = '⚡';

            html += `
                <div class="advisor">
                    <div class="advisor__icon">${icon}</div>
                    <div class="advisor__name" style="font-size: 0.65rem; word-break: break-word; line-height: 1.1; margin-bottom: 4px;">${name.replace('Advisor', '')}</div>
                    <div class="advisor__vote ${vClass}">${iconStr}</div>
                    <div class="advisor__weight">w=${w}</div>
                </div>
            `;
        }
        grid.innerHTML = html;
        // Dynamically adjust grid columns if there are more than 5 advisors
        const numAdvisors = Object.keys(votes).length;
        if (numAdvisors > 5) {
            grid.style.gridTemplateColumns = `repeat(${numAdvisors}, 1fr)`;
        }
    }

    // Update consensus bar
    const score = data.consensus || 0;
    $('consensusFill').style.width = `${score * 100}%`;
    $('consensusScore').textContent = `${(score * 100).toFixed(1)}% — ${data.approved ? 'APPROVED' : 'REJECTED'}`;
    $('consensusScore').style.color = data.approved ? 'var(--green)' : 'var(--red)';

    $('councilBadge').textContent = data.summary || `${Object.keys(votes).length} Advisors`;
}

function updateEnsemble(data) {
    state.lastEnsemble = data;

    const trending = data.trending || 0.5;
    const ranging = data.ranging || 0.5;
    const agreement = data.agreement || 0;

    $('trendingBar').style.width = `${trending * 100}%`;
    $('trendingProb').textContent = (trending * 100).toFixed(1) + '%';

    $('rangingBar').style.width = `${ranging * 100}%`;
    $('rangingProb').textContent = (ranging * 100).toFixed(1) + '%';

    $('agreementBar').style.width = `${agreement * 100}%`;
    $('agreementPct').textContent = (agreement * 100).toFixed(0) + '%';
}

function updatePosition(data) {
    state.position = data;
    const panel = $('positionBody');
    const badge = $('positionBadge');

    const dir = data.direction || 'LONG';
    const isLong = (dir === 'LONG' || dir === 'BUY');
    badge.innerHTML = `<span class="signal-badge signal-badge--${isLong ? 'buy' : 'sell'}">${isLong ? '🟢 BUY' : '🔴 SELL'} ${(data.symbol || '').toUpperCase()}</span>`;
    badge.className = ""; // clear default padding

    let trailHtml = '';
    if (data.trail_phase && data.trail_phase !== 'INITIAL') {
        const phaseColor = data.trail_phase === 'BREAKEVEN' ? '#F59E0B' : '#10B981';
        trailHtml = `<div class="position-hud__trail" style="color: ${phaseColor}; border-color: ${phaseColor}40; background: ${phaseColor}15;">⚡ ${data.trail_phase}</div>`;
    }

    panel.innerHTML = `
        <div class="position-hud">
            <div class="position-hud__metrics">
                <div class="signal-metric">
                    <span class="signal-metric__label">ENTRY PRICE</span>
                    <span class="signal-metric__value">${state.currency || '$'}${(data.entry_price || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
                </div>
                <div class="signal-metric" style="text-align: right;">
                    <span class="signal-metric__label">CURRENT PRICE</span>
                    <span class="signal-metric__value" id="positionCurrentPrice">${state.currency || '$'}${(data.current_price || data.entry_price || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
                </div>
            </div>

            <div class="position-hud__visual">
                <div class="position-hud__target position-hud__target--sl">
                    <span class="position-hud__target-label">SL</span>
                    <span class="position-hud__target-price">${state.currency || '$'}${(data.stop_loss || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
                </div>
                <div class="position-hud__line">
                    <div class="position-hud__glow-dot"></div>
                </div>
                <div class="position-hud__target position-hud__target--tp">
                    <span class="position-hud__target-label">TP</span>
                    <span class="position-hud__target-price">${state.currency || '$'}${(data.take_profit || 0).toLocaleString(undefined, {minimumFractionDigits: 2})}</span>
                </div>
            </div>
            ${trailHtml}

            <div class="position-hud__pnl-container">
                <div class="position-hud__pnl-label">UNREALIZED P&L</div>
                <div class="position-hud__pnl" id="positionPnl">${state.currency || '$'}0.00</div>
            </div>
        </div>
    `;
}

function updateTick(data) {
    const livePriceEl = $('livePrice');
    if (livePriceEl && data.current_price) {
        livePriceEl.innerHTML = `LIVE PRICE: ${state.currency || '$'}${data.current_price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
    }

    if (state.position) {
        // Update PnL
        const pnl = data.unrealized_pnl || 0;
        const pnlEl = document.getElementById('positionPnl');
        if (pnlEl) {
            pnlEl.textContent = formatCurrency(pnl, true);
            pnlEl.className = `position-hud__pnl ${pnl >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update Current Price
        if (data.current_price) {
            const priceEl = document.getElementById('positionCurrentPrice');
            if (priceEl) {
                priceEl.textContent = `${state.currency || '$'}${data.current_price.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            }
            
            // Animate Glow Dot
            if (state.position.stop_loss && state.position.take_profit) {
                const sl = state.position.stop_loss;
                const tp = state.position.take_profit;
                const current = data.current_price;
                
                const dot = document.querySelector('.position-hud__glow-dot');
                if (dot) {
                    const totalRange = Math.abs(tp - sl);
                    if (totalRange > 0) {
                        let pct = Math.abs(current - sl) / totalRange;
                        pct = Math.max(0, Math.min(1, pct));
                        dot.style.left = `${pct * 100}%`;
                    }
                }
            }
        }
    }
}

function closeTrade(data) {
    state.position = null;
    state.totalTrades++;
    if (data.pnl > 0) state.winningTrades++;
    state.totalPnl += data.pnl || 0;
    state.balance += data.pnl || 0;

    // Reset position panel
    $('positionBody').innerHTML = `
        <div class="position-empty">
            <span class="position-empty__icon">⏳</span>
            <span>Waiting for signal...</span>
        </div>
    `;
    $('positionBadge').textContent = 'No Position';
    $('positionBadge').style.color = '';

    // Add to trade history
    addTradeCard(data);
    refreshMetrics();
}

function addTradeCard(data) {
    const history = $('tradeHistory');
    const empty = history.querySelector('.trade-empty');
    if (empty) empty.remove();

    const pnl = data.pnl || 0;
    const icon = pnl >= 0 ? '💰' : '🔻';
    const reason = data.exit_reason || 'signal';

    const card = document.createElement('div');
    card.className = 'trade-card';
    card.innerHTML = `
        <div class="trade-card__icon">${icon}</div>
        <div class="trade-card__info">
            <div class="trade-card__pair">
                ${(data.symbol || 'BTCUSDT').toUpperCase()}
                <span style="font-size: 0.65rem; color: var(--text-muted)">${data.side || 'SHORT'}</span>
            </div>
            <div class="trade-card__meta">
                ${state.currency || '$'}${(data.entry_price || 0).toFixed(2)} → ${state.currency || '$'}${(data.exit_price || 0).toFixed(2)}
                · ${Math.round(data.duration_seconds || 0)}s
            </div>
        </div>
        <div>
            <div class="trade-card__pnl ${pnl >= 0 ? 'positive' : 'negative'}">${formatCurrency(pnl, true)}</div>
            <div class="trade-card__reason">${reason}</div>
        </div>
    `;

    history.insertBefore(card, history.firstChild);
    $('tradeBadge').textContent = `${state.totalTrades} trades`;
}

// ── Utilities ────────────────────────────────────────────────────────────────

function formatCurrency(val, signed = false) {
    const abs = Math.abs(val);
    const curr = state.currency || '$';
    const formatted = curr + abs.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if (signed) {
        return val >= 0 ? '+' + formatted : '-' + formatted;
    }
    return formatted;
}

function updateUptime() {
    const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    $('uptime').textContent = `${h}:${m}:${s}`;
}

// ── Simulation Mode ──────────────────────────────────────────────────────────
// When the server isn't running, simulate realistic data for demo purposes

function startSimulation() {
    updateConnectionStatus(true);
    $('wsStatusText').textContent = 'Simulation';

    const regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'QUIET'];
    let candleCount = 0;

    // Simulate candle ticks
    setInterval(() => {
        candleCount++;
        state.candles = candleCount;

        // Random regime
        if (candleCount % 5 === 0) {
            state.regime = regimes[Math.floor(Math.random() * regimes.length)];
            state.adx = 15 + Math.random() * 35;
        }

        // Simulate ensemble
        const trending = 0.1 + Math.random() * 0.8;
        const ranging = 0.1 + Math.random() * 0.8;
        const agreement = 1 - Math.abs(trending - ranging);
        updateEnsemble({ trending, ranging, agreement });

        // Simulate council
        const advisors = ['Momentum', 'Structure', 'Volume', 'Regime', 'Risk'];
        const voteOptions = ['APPROVE', 'REJECT', 'ABSTAIN'];
        const votes = {};
        let approveWeight = 0, rejectWeight = 0;
        const weights = { Momentum: 1.5, Structure: 1.2, Volume: 1.0, Regime: 1.3, Risk: 2.0 };

        advisors.forEach(name => {
            const vote = voteOptions[Math.floor(Math.random() * 3)];
            votes[name] = { vote, conviction: 0.3 + Math.random() * 0.7 };
            if (vote === 'APPROVE') approveWeight += weights[name] * votes[name].conviction;
            else if (vote === 'REJECT') rejectWeight += weights[name] * votes[name].conviction;
        });

        const total = approveWeight + rejectWeight;
        const consensus = total > 0 ? approveWeight / total : 0;
        const approved = consensus >= 0.6;
        const approves = Object.values(votes).filter(v => v.vote === 'APPROVE').length;
        const rejects = Object.values(votes).filter(v => v.vote === 'REJECT').length;
        const abstains = Object.values(votes).filter(v => v.vote === 'ABSTAIN').length;

        updateCouncil({
            votes,
            consensus,
            approved,
            summary: `${approves}A/${rejects}R/${abstains}S`,
        });

        // Occasionally generate a signal
        if (candleCount % 3 === 0) {
            const dir = Math.random() > 0.5 ? 'BUY' : 'SELL';
            const price = 73000 + Math.random() * 1000;
            const blocked = !approved;

            addSignal({
                direction: dir,
                symbol: 'btcusdt',
                price: price,
                conviction: 0.15 + Math.random() * 0.7,
                council_score: consensus,
                blocked: blocked,
                reason: `${state.regime} regime | council=${(consensus*100).toFixed(0)}%`,
            });

            if (!blocked && !state.position) {
                updatePosition({
                    direction: dir === 'BUY' ? 'LONG' : 'SHORT',
                    symbol: 'btcusdt',
                    entry_price: price,
                    current_price: price,
                    stop_loss: dir === 'BUY' ? price - 50 : price + 50,
                    take_profit: dir === 'BUY' ? price + 100 : price - 100,
                });
            }
        }

        // Simulate trade close
        if (state.position && Math.random() > 0.85) {
            const pnl = (Math.random() - 0.4) * 200;
            closeTrade({
                symbol: 'btcusdt',
                side: state.position.direction,
                entry_price: state.position.entry_price,
                exit_price: state.position.entry_price + pnl / 4,
                pnl: pnl,
                pnl_pct: pnl / state.position.entry_price,
                exit_reason: pnl > 0 ? 'take_profit' : 'stop_loss',
                duration_seconds: 30 + Math.random() * 300,
            });
        }

        // Update position P&L
        if (state.position) {
            const noise = (Math.random() - 0.5) * 80;
            updateTick({ unrealized_pnl: noise });
        }

        refreshMetrics();
    }, 2000);
}

// ── Initialize ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setInterval(updateUptime, 1000);

    // Try connecting to live server, fall back to simulation
    try {
        connectWebSocket();
    } catch {
        startSimulation();
    }

    // Start simulation after 3s if not connected
    setTimeout(() => {
        if (!state.connected) {
            startSimulation();
        }
    }, 3000);
});

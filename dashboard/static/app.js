/**
 * AdaRAG-Ops Autonomous Technical Support Dashboard — Client Engine
 * Handles WebSocket communication, terminal streaming, node graph visualization,
 * command history, and cache telemetry.
 */

(function () {
  'use strict';

  // State
  let ws = null;
  let isExecuting = false;
  let autoScroll = true;
  let commandHistory = [];
  let historyIndex = -1;
  let activeNodes = new Set();
  let sessionStartTime = Date.now();

  // DOM Elements
  const terminalScreen = document.getElementById('terminal-screen');
  const messagesStream = document.getElementById('terminal-messages-stream');
  const terminalInput = document.getElementById('terminal-input');
  const terminalForm = document.getElementById('terminal-input-form');
  const liveWorking = document.getElementById('term-live-working');
  const liveStatusText = document.getElementById('term-live-status-text');
  const btnClear = document.getElementById('btn-clear-term');
  const btnCopy = document.getElementById('btn-copy-term');
  const btnAutoScroll = document.getElementById('btn-autoscroll');
  const winClose = document.getElementById('win-btn-close');

  // Status Labels
  const lblCacheCount = document.getElementById('lbl-cache-count');
  const lblKbCount = document.getElementById('lbl-kb-count');
  const lblCacheDbRecords = document.getElementById('lbl-cache-db-records');
  const chkCacheToggle = document.getElementById('chk-cache-toggle');
  const btnQuickCacheClear = document.getElementById('btn-quick-cache-clear');
  const btnQuickCacheView = document.getElementById('btn-quick-cache-view');
  const pipelineBadge = document.getElementById('pipeline-active-badge');

  // Telemetry Elements
  const valLatency = document.getElementById('val-telemetry-latency');
  const valCost = document.getElementById('val-telemetry-cost');
  const valSavedCalls = document.getElementById('val-saved-calls');
  const valHitRate = document.getElementById('val-telemetry-hitrate');
  const valTotalHits = document.getElementById('val-total-hits');
  const valTotalLookups = document.getElementById('val-total-lookups');
  const valConfidence = document.getElementById('val-telemetry-confidence');

  // Modals
  const modalCache = document.getElementById('modal-cache');
  const modalKb = document.getElementById('modal-kb');
  const modalPresets = document.getElementById('modal-presets');
  const modalStatus = document.getElementById('modal-status');

  // --------------------------------------------------------------------------
  // WebSocket Connection
  // --------------------------------------------------------------------------
  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/terminal`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
      console.log('[AdaRAG-Ops] WebSocket connected.');
      document.getElementById('status-engine-dot').className = 'status-indicator-dot online';
      document.getElementById('lbl-engine-status').textContent = 'Online';
    };

    ws.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.onclose = function () {
      console.warn('[AdaRAG-Ops] WebSocket closed. Reconnecting in 2s...');
      document.getElementById('status-engine-dot').className = 'status-indicator-dot';
      document.getElementById('lbl-engine-status').textContent = 'Reconnecting...';
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = function (err) {
      console.error('[AdaRAG-Ops] WS Error:', err);
    };
  }

  // --------------------------------------------------------------------------
  // Message Handling
  // --------------------------------------------------------------------------
  function handleServerMessage(data) {
    switch (data.type) {
      case 'terminal_init':
        updateCacheCountUI(data.cache_records);
        chkCacheToggle.checked = !!data.cache_enabled;
        break;

      case 'terminal_command_result':
        renderTerminalCommandOutput(data.command, data.output, data.level);
        if (data.cache_metrics) {
          updateTelemetryMetrics(data.cache_metrics);
        }
        break;

      case 'terminal_clear':
        clearTerminalScreen();
        break;

      case 'pipeline_start':
        isExecuting = true;
        resetPipelineTopology();
        setLiveWorking(true, `Multi-agent AI pipeline analyzing incident...`);
        pipelineBadge.textContent = 'EXECUTING';
        pipelineBadge.style.color = 'var(--text-cyan)';
        pipelineBadge.style.borderColor = 'var(--accent-cyan)';
        break;

      case 'step_update':
        renderStepBlock(data.node);
        updateNodeTopology(data.node);
        break;

      case 'pipeline_complete':
        isExecuting = false;
        setLiveWorking(false);
        renderFinalResolution(data);
        if (data.cache_metrics) {
          updateTelemetryMetrics(data.cache_metrics);
        }
        valLatency.textContent = `${data.elapsed_sec.toFixed(2)}s`;
        pipelineBadge.textContent = data.escalated ? 'ESCALATED' : (data.cache_hit ? 'CACHE HIT' : 'RESOLVED');
        pipelineBadge.style.color = data.escalated ? 'var(--text-red)' : 'var(--text-green)';
        pipelineBadge.style.borderColor = data.escalated ? 'var(--accent-red)' : 'var(--accent-green)';
        break;

      case 'pipeline_error':
        isExecuting = false;
        setLiveWorking(false);
        renderErrorBlock(data.error);
        pipelineBadge.textContent = 'ERROR';
        pipelineBadge.style.color = 'var(--text-red)';
        break;
    }

    if (autoScroll) {
      scrollToBottom();
    }
  }

  // --------------------------------------------------------------------------
  // Rendering Helpers
  // --------------------------------------------------------------------------
  function echoUserQuery(text) {
    const entry = document.createElement('div');
    entry.className = 'terminal-user-entry';
    entry.innerHTML = `
      <span class="terminal-user-prompt">ops@support-ai:~$</span>
      <span class="terminal-user-text">${escapeHtml(text)}</span>
    `;
    messagesStream.appendChild(entry);
    if (autoScroll) scrollToBottom();
  }

  function renderTerminalCommandOutput(command, output, level) {
    const box = document.createElement('div');
    box.className = 'term-step-block';
    
    let colorClass = '#0284c7';
    let bgTint = '#f0f9ff';
    if (level === 'success') { colorClass = '#059669'; bgTint = '#f0fdf4'; }
    if (level === 'warning') { colorClass = '#d97706'; bgTint = '#fffbeb'; }
    if (level === 'error') { colorClass = '#dc2626'; bgTint = '#fef2f2'; }

    box.style.borderLeft = `4px solid ${colorClass}`;
    box.style.background = bgTint;

    box.innerHTML = `
      <div class="term-step-header" style="background:transparent; border-bottom:1px solid var(--border-color);">
        <div class="term-step-label">
          <span>⚡ Command: <strong>${escapeHtml(command)}</strong></span>
        </div>
        <span class="term-step-elapsed">${new Date().toLocaleTimeString()}</span>
      </div>
      <div class="term-step-body" style="font-family:var(--font-mono); white-space:pre-wrap; font-size:12px; line-height:1.6; color:var(--text-main); font-weight:500;">${escapeHtml(output)}</div>
    `;
    messagesStream.appendChild(box);
  }

  function renderStepBlock(node) {
    const block = document.createElement('div');
    block.className = 'term-step-block' + (node.hit ? ' cache-hit' : '');

    let badgeClass = 'badge-cyan';
    let badgeText = `Step ${node.step_index}`;

    if (node.node_name === 'cache_lookup') {
      badgeClass = node.hit ? 'badge-green' : 'badge-amber';
      badgeText = node.hit ? `CACHE HIT [${(node.match_type || 'EXACT').toUpperCase()}]` : 'CACHE MISS';
    } else if (node.node_name === 'classify') {
      badgeClass = node.severity === 'L3' ? 'badge-red' : (node.severity === 'L2' ? 'badge-amber' : 'badge-green');
      badgeText = `${node.severity} • ${node.category}`;
    } else if (node.node_name === 'router') {
      badgeClass = 'badge-purple';
      badgeText = `${node.chosen_tier} • ${node.chosen_model}`;
    } else if (node.node_name === 'verify') {
      badgeClass = node.verified ? 'badge-green' : 'badge-amber';
      badgeText = node.verified ? 'SAFE & VERIFIED' : 'NEEDS HUMAN REVIEW';
    }

    let bodyHtml = '';

    if (node.node_name === 'cache_lookup') {
      if (node.hit) {
        bodyHtml = `
          <div class="term-data-row">
            <span class="term-key">Cache Status:</span>
            <span class="term-val" style="color:var(--text-green); font-weight:800;">⚡ Verified Cache HIT [${(node.match_type || 'EXACT').toUpperCase()}]</span>
          </div>
          <div class="term-data-row">
            <span class="term-key">Lookup Speed:</span>
            <span class="term-val"><strong>${(node.lookup_latency * 1000).toFixed(1)}ms</strong> (${node.lookup_latency.toFixed(4)}s) | Match: <strong>${(node.similarity * 100).toFixed(1)}%</strong></span>
          </div>
          <div class="term-data-row">
            <span class="term-key">Record ID:</span>
            <span class="term-val" style="font-family:var(--font-mono); color:var(--text-cyan); font-weight:700;">${node.record_id || 'CACHE_HIT'}</span>
          </div>
          <div class="term-data-row">
            <span class="term-key">Cost Benefit:</span>
            <span class="term-val" style="color:var(--text-green); font-weight:700;">$0.00000 (0 LLM API calls, 100% cost avoided)</span>
          </div>
        `;
      } else {
        bodyHtml = `
          <div class="term-data-row">
            <span class="term-key">Cache Status:</span>
            <span class="term-val" style="color:var(--text-amber); font-weight:700;">Cache Miss (${node.lookup_latency.toFixed(4)}s)</span>
          </div>
          <div class="term-data-row">
            <span class="term-key">Action:</span>
            <span class="term-val">Executing live multi-agent RAG pipeline (Classifier → Knowledge Search → Model Router → RCA → Resolution → Verification).</span>
          </div>
        `;
      }
    } else if (node.node_name === 'classify') {
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">Category:</span>
          <span class="term-val" style="color:var(--text-purple); font-weight:600;">${capitalize(node.category)}</span>
        </div>
        <div class="term-data-row">
          <span class="term-key">Priority:</span>
          <span class="term-val" style="color:${node.severity === 'L3' ? 'var(--text-red)' : (node.severity === 'L2' ? 'var(--text-amber)' : 'var(--text-green)')}; font-weight:700;">${node.severity}</span>
        </div>
        ${node.summary ? `
          <div class="term-data-row">
            <span class="term-key">Summary:</span>
            <span class="term-val">${escapeHtml(node.summary)}</span>
          </div>
        ` : ''}
      `;
    } else if (node.node_name === 'knowledge_search') {
      const sourcesList = (node.sources || []).map(s => `<span style="color:var(--text-green); background:rgba(16,185,129,0.1); padding:2px 6px; border-radius:4px; font-size:11px;">📄 ${escapeHtml(s)}</span>`).join(' ');
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">FAISS Retrieval:</span>
          <span class="term-val">Found <strong>${node.chunks_count || 0}</strong> relevant chunks across documentation:</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:4px;">${sourcesList || 'None found'}</div>
      `;
    } else if (node.node_name === 'evidence_assessment') {
      const missingTxt = (node.missing_evidence && node.missing_evidence.length > 0)
        ? node.missing_evidence.map(m => `<span style="color:var(--text-amber);">⚠️ ${escapeHtml(m)}</span>`).join(', ')
        : `<span style="color:var(--text-green);">Complete evidence grounding</span>`;
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">Retrieval Conf:</span>
          <span class="term-val" style="color:var(--text-green); font-weight:700;">${node.retrieval_confidence || 0}% (Top-1 Sim: ${node.top_1_similarity || 0})</span>
        </div>
        <div class="term-data-row">
          <span class="term-key">Telemetry Audit:</span>
          <span class="term-val">${missingTxt}</span>
        </div>
      `;
    } else if (node.node_name === 'router') {
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">Routed Model:</span>
          <span class="term-val" style="color:var(--text-cyan); font-weight:700;">${escapeHtml(node.chosen_model)} <span style="font-size:11px; color:var(--text-purple);">[${node.chosen_tier}]</span></span>
        </div>
        <div class="term-data-row">
          <span class="term-key">Provider:</span>
          <span class="term-val" style="color:var(--text-amber);">${escapeHtml(node.provider)}</span>
        </div>
        <div class="term-data-row">
          <span class="term-key">Complexity / Conf:</span>
          <span class="term-val">Score: <strong>${node.complexity_score}</strong> | Retrieval Confidence: <strong style="color:var(--text-green);">${node.retrieval_confidence}%</strong></span>
        </div>
      `;
    } else if (node.node_name === 'diagnose') {
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">Confidence:</span>
          <span class="term-val" style="font-weight:700; color:${node.confidence === 'high' ? 'var(--text-green)' : (node.confidence === 'medium' ? 'var(--text-amber)' : 'var(--text-red)')};">${capitalize(node.confidence)}</span>
        </div>
        <div style="margin-top:6px; color:var(--text-body); font-weight:500; line-height:1.6;">${escapeHtml(node.root_cause || 'Analyzing root cause...')}</div>
      `;
    } else if (node.node_name === 'resolve') {
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">Remediation Status:</span>
          <span class="term-val" style="color:${node.resolution_status === 'resolved' ? 'var(--text-green)' : 'var(--text-amber)'}; font-weight:700;">${node.resolution_status === 'resolved' ? 'Solution Ready' : 'Needs Human Review'}</span>
        </div>
      `;
    } else if (node.node_name === 'verify') {
      bodyHtml = `
        <div class="term-data-row">
          <span class="term-key">Verification:</span>
          <span class="term-val" style="color:${node.verified ? 'var(--text-green)' : 'var(--text-amber)'}; font-weight:700;">${node.verified ? 'Solution verified as safe and complete.' : 'Flagged for human review.'}</span>
        </div>
        ${node.verification_notes ? `
          <div class="term-data-row">
            <span class="term-key">Notes:</span>
            <span class="term-val">${escapeHtml(node.verification_notes)}</span>
          </div>
        ` : ''}
        <div class="term-data-row">
          <span class="term-key">Calibrated Conf:</span>
          <span class="term-val" style="color:var(--text-amber); font-weight:700;">${node.calibrated_confidence || 85}%</span>
        </div>
      `;
    }

    block.innerHTML = `
      <div class="term-step-header">
        <div class="term-step-label">
          <span>>></span>
          <span>${escapeHtml(node.label)}</span>
          <span class="term-step-badge ${badgeClass}">${escapeHtml(badgeText)}</span>
        </div>
        <span class="term-step-elapsed">+${node.elapsed_sec}s</span>
      </div>
      <div class="term-step-body">${bodyHtml}</div>
    `;

    messagesStream.appendChild(block);
  }

  function renderFinalResolution(data) {
    const card = document.createElement('div');
    card.className = 'term-final-resolution' + (data.escalated ? ' escalated' : '');

    const formattedAnswer = formatMarkdownToHtml(data.final_answer);

    let cacheBenefitText = '';
    if (data.escalated) {
      cacheBenefitText = `[i] Safety Policy: Escalated/unverified tickets are NOT saved to cache. Only confirmed resolutions are cached.`;
    } else if (data.cache_hit) {
      cacheBenefitText = `[i] Cache Benefit: Served instantly from cache [${(data.cache_match_type || 'EXACT').toUpperCase()}] with 0 LLM calls and $0.00 model cost.`;
    } else {
      cacheBenefitText = `[i] Cache Persisted: Verified resolution saved to cache for future 0-cost, sub-20ms retrieval.`;
    }

    card.innerHTML = `
      <div class="term-resolution-header">
        <div class="term-resolution-title" style="color:${data.escalated ? 'var(--text-red)' : 'var(--text-green)'};">
          <span>${data.escalated ? '⚠️' : '✅'}</span>
          <span>${data.escalated ? 'ESCALATED  --  Human SRE Review Required' : 'RESOLVED  --  Actionable Incident Resolution'}</span>
        </div>
        <span style="font-size:12px; font-family:var(--font-mono); color:var(--text-dim);">Duration: <strong>${data.elapsed_sec}s</strong></span>
      </div>

      <div class="resolution-text-wrap" id="resolution-content-${Date.now()}">
        ${formattedAnswer}
      </div>

      <div class="resolution-meta-footer">
        <div><strong>Model:</strong> ${escapeHtml(data.model_used || 'Cache Guard')}</div>
        <div><strong>Category:</strong> ${capitalize(data.category)}</div>
        <div><strong>Priority:</strong> ${data.severity}</div>
        <div style="color:${data.escalated ? 'var(--text-amber)' : 'var(--text-green)'}; width:100%; margin-top:4px;">
          ${escapeHtml(cacheBenefitText)}
        </div>
      </div>
    `;

    messagesStream.appendChild(card);

    // Attach copy button handlers
    card.querySelectorAll('.btn-copy-code').forEach(btn => {
      btn.addEventListener('click', function () {
        const pre = this.closest('pre');
        const codeText = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText.replace('Copy', '');
        navigator.clipboard.writeText(codeText.trim());
        const orig = this.textContent;
        this.textContent = 'Copied!';
        setTimeout(() => this.textContent = orig, 1500);
      });
    });
  }

  function renderErrorBlock(err) {
    const block = document.createElement('div');
    block.className = 'term-step-block';
    block.style.borderColor = 'var(--accent-red)';
    block.innerHTML = `
      <div class="term-step-header" style="background:rgba(239, 68, 68, 0.1);">
        <div class="term-step-label" style="color:var(--text-red);">
          <span>❌ Execution Error</span>
        </div>
      </div>
      <div class="term-step-body" style="color:#fca5a5;">${escapeHtml(err)}</div>
    `;
    messagesStream.appendChild(block);
  }

  // --------------------------------------------------------------------------
  // Pipeline Topology Dynamic Visualizer
  // --------------------------------------------------------------------------
  function resetPipelineTopology() {
    activeNodes.clear();
    const items = document.querySelectorAll('.pipe-node-item');
    items.forEach(el => {
      el.className = 'pipe-node-item';
      const timeSpan = el.querySelector('.pipe-node-time');
      if (timeSpan) timeSpan.textContent = '--';
    });
  }

  function updateNodeTopology(node) {
    const el = document.getElementById(`node-item-${node.node_name}`);
    if (!el) return;

    // Previous active nodes become completed (unless special cache-hit or bypassed)
    activeNodes.forEach(name => {
      const prevEl = document.getElementById(`node-item-${name}`);
      if (prevEl && prevEl !== el && !prevEl.classList.contains('cache-hit-node') && !prevEl.classList.contains('bypassed')) {
        prevEl.classList.remove('running');
        prevEl.classList.add('completed');
      }
    });

    activeNodes.add(node.node_name);
    el.classList.add('running');

    if (node.node_name === 'cache_lookup' && node.hit) {
      el.classList.remove('running');
      el.classList.add('cache-hit-node');

      // Systematic transparency: Mark intermediate LLM pipeline stages as bypassed at 0 cost
      const intermediateNodes = [
        'classify',
        'knowledge_search',
        'evidence_assessment',
        'router',
        'diagnose',
        'resolve',
        'verify'
      ];
      intermediateNodes.forEach(name => {
        const bypassEl = document.getElementById(`node-item-${name}`);
        if (bypassEl) {
          bypassEl.className = 'pipe-node-item bypassed';
          const tSpan = bypassEl.querySelector('.pipe-node-time');
          if (tSpan) tSpan.textContent = '0.00s';
          const dSpan = bypassEl.querySelector('.pipe-node-detail');
          if (dSpan) dSpan.innerHTML = '<span style="color:var(--text-green); font-weight:700;">⚡ Bypassed by Cache Guard ($0.00)</span>';
        }
      });
    }

    const timeSpan = el.querySelector('.pipe-node-time');
    if (timeSpan) timeSpan.textContent = `+${node.elapsed_sec}s`;

    const detailSpan = el.querySelector('.pipe-node-detail');
    if (detailSpan) {
      if (node.node_name === 'cache_lookup') {
        detailSpan.innerHTML = node.hit 
          ? `<strong style="color:var(--text-green);">HIT [${(node.match_type || 'EXACT').toUpperCase()}]</strong> • Saved ${node.cost_saved || '$0.005'}` 
          : 'Cache MISS • Running multi-agent RAG';
      } else if (node.node_name === 'classify') {
        detailSpan.textContent = `${node.category.toUpperCase()} • ${node.severity}`;
      } else if (node.node_name === 'router') {
        detailSpan.textContent = `${node.chosen_model} [${node.chosen_tier}]`;
      } else if (node.node_name === 'verify') {
        detailSpan.textContent = node.verified ? 'Verified Safe (100%)' : 'Flagged for Human Review';
      } else if (node.node_name === 'close') {
        detailSpan.innerHTML = node.hit || activeNodes.has('cache_lookup') 
          ? '<strong style="color:var(--text-green);">⚡ 0-Cost Cache Resolution Replay</strong>' 
          : 'Resolution Verified & Persisted';
      }
    }

    setLiveWorking(true, `[${node.step_index}/6] ${node.label}...`);
  }

  // --------------------------------------------------------------------------
  // Telemetry HUD Update
  // --------------------------------------------------------------------------
  function updateTelemetryMetrics(m) {
    if (!m) return;
    valCost.textContent = `$${(m.avoided_llm_cost_usd || 0).toFixed(4)}`;
    valSavedCalls.textContent = m.avoided_model_calls || 0;
    valHitRate.textContent = `${(m.hit_rate_percent || 0).toFixed(1)}%`;
    valTotalHits.textContent = (m.exact_hits || 0) + (m.semantic_hits || 0);
    valTotalLookups.textContent = m.total_lookups || 0;
    updateCacheCountUI(m.total_records_in_db || 0);
  }

  function updateCacheCountUI(count) {
    if (lblCacheCount) lblCacheCount.textContent = `${count} Records`;
    if (lblCacheDbRecords) lblCacheDbRecords.textContent = `${count} in DB`;
  }

  function setLiveWorking(visible, text) {
    if (!liveWorking) return;
    liveWorking.style.display = visible ? 'flex' : 'none';
    if (text && liveStatusText) {
      liveStatusText.textContent = text;
    }
  }

  function scrollToBottom() {
    terminalScreen.scrollTop = terminalScreen.scrollHeight;
  }

  function clearTerminalScreen() {
    messagesStream.innerHTML = '';
    resetPipelineTopology();
    pipelineBadge.textContent = 'STANDBY';
    pipelineBadge.style.color = 'var(--text-green)';
    pipelineBadge.style.borderColor = 'rgba(16, 185, 129, 0.3)';
  }

  // --------------------------------------------------------------------------
  // User Input Submission & Command History
  // --------------------------------------------------------------------------
  window.submitTerminalQuery = function (explicitQuery) {
    const raw = (explicitQuery !== undefined ? explicitQuery : terminalInput.value).trim();
    if (!raw) return;

    if (!explicitQuery) {
      terminalInput.value = '';
    }

    // Add to command history
    commandHistory.push(raw);
    historyIndex = commandHistory.length;

    echoUserQuery(raw);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: raw }));
    } else {
      renderErrorBlock('WebSocket disconnected. Please wait for reconnection or refresh.');
    }
  };

  // Keyboard navigation for command history
  terminalInput.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (commandHistory.length > 0 && historyIndex > 0) {
        historyIndex--;
        terminalInput.value = commandHistory[historyIndex];
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIndex < commandHistory.length - 1) {
        historyIndex++;
        terminalInput.value = commandHistory[historyIndex];
      } else {
        historyIndex = commandHistory.length;
        terminalInput.value = '';
      }
    } else if (e.key === 'l' && e.ctrlKey) {
      e.preventDefault();
      clearTerminalScreen();
    }
  });

  // Action Buttons
  btnClear.addEventListener('click', clearTerminalScreen);
  winClose.addEventListener('click', clearTerminalScreen);

  btnCopy.addEventListener('click', function () {
    const text = messagesStream.innerText;
    navigator.clipboard.writeText(text);
    btnCopy.textContent = 'Copied!';
    setTimeout(() => btnCopy.innerHTML = '<span>📋</span> Copy Log', 1500);
  });

  btnAutoScroll.addEventListener('click', function () {
    autoScroll = !autoScroll;
    btnAutoScroll.classList.toggle('active', autoScroll);
  });

  // Quick Action Pills
  document.querySelectorAll('.quick-pill').forEach(pill => {
    pill.addEventListener('click', function () {
      const query = this.getAttribute('data-query');
      const cmd = this.getAttribute('data-command');
      window.submitTerminalQuery(query || cmd);
    });
  });

  // Cache Toggle Switch
  chkCacheToggle.addEventListener('change', async function () {
    try {
      const res = await fetch('/api/cache/toggle', { method: 'POST' });
      const data = await res.json();
      const statusText = data.cache_enabled ? 'ENABLED' : 'DISABLED';
      renderTerminalCommandOutput('cache-toggle', `[OK] Incident Response Cache is now ${statusText}.`, 'info');
    } catch (e) {
      console.error(e);
    }
  });

  // Cache Clear Button
  btnQuickCacheClear.addEventListener('click', async function () {
    if (!confirm('Are you sure you want to purge all persistent cache records in SQLite?')) return;
    try {
      const res = await fetch('/api/cache/clear', { method: 'POST' });
      const data = await res.json();
      renderTerminalCommandOutput('cache-clear', `[OK] Purged ${data.cleared_count} records from cache.`, 'warning');
      loadCacheTable();
    } catch (e) {
      console.error(e);
    }
  });

  // --------------------------------------------------------------------------
  // Modals Management
  // --------------------------------------------------------------------------
  function openModal(modal) {
    if (!modal) return;
    modal.classList.add('active');
  }

  function closeModal(modal) {
    if (!modal) return;
    modal.classList.remove('active');
  }

  document.querySelectorAll('[data-close]').forEach(btn => {
    btn.addEventListener('click', function () {
      const modalId = this.getAttribute('data-close');
      closeModal(document.getElementById(modalId));
    });
  });

  document.querySelectorAll('.modal-backdrop').forEach(modal => {
    modal.addEventListener('click', function (e) {
      if (e.target === this) closeModal(this);
    });
  });

  document.getElementById('btn-open-cache-modal').addEventListener('click', () => {
    openModal(modalCache);
    loadCacheTable();
  });
  btnQuickCacheView.addEventListener('click', () => {
    openModal(modalCache);
    loadCacheTable();
  });

  document.getElementById('btn-open-kb-modal').addEventListener('click', () => {
    openModal(modalKb);
    loadKbDocuments();
  });

  document.getElementById('btn-open-presets').addEventListener('click', () => {
    openModal(modalPresets);
    loadPresetsList();
  });

  document.getElementById('btn-open-status-modal').addEventListener('click', () => {
    openModal(modalStatus);
    loadSystemStatus();
  });

  // --------------------------------------------------------------------------
  // API Fetchers
  // --------------------------------------------------------------------------
  async function loadCacheTable() {
    const tbody = document.getElementById('tbody-cache-records');
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;">Loading...</td></tr>`;
    try {
      const res = await fetch('/api/cache/records');
      const data = await res.json();
      if (!data.records || data.records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-dim);">No cached records found.</td></tr>`;
        return;
      }
      tbody.innerHTML = data.records.map(r => `
        <tr>
          <td><span style="color:var(--text-purple); font-weight:600;">${escapeHtml(r.affected_service || 'system')}</span></td>
          <td style="max-width:280px; word-break:break-word;">${escapeHtml(r.query)}</td>
          <td><span class="term-step-badge ${r.severity === 'L3' ? 'badge-red' : (r.severity === 'L2' ? 'badge-amber' : 'badge-green')}">${r.severity}</span></td>
          <td style="font-family:var(--font-mono); color:var(--text-green); font-weight:700;">${r.hit_count}</td>
          <td style="font-size:11px; color:var(--text-dim);">${escapeHtml(r.model_id || 'cache')}</td>
          <td>
            <button class="btn-term-action" style="color:var(--accent-cyan);" onclick="window.submitTerminalQuery('${escapeJs(r.query)}'); document.getElementById('modal-cache').classList.remove('active');">
              ▶ Re-run
            </button>
          </td>
        </tr>
      `).join('');
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="6" style="color:var(--text-red);">Failed to load cache records: ${escapeHtml(e.message)}</td></tr>`;
    }
  }

  document.getElementById('btn-refresh-cache-table').addEventListener('click', loadCacheTable);

  async function loadKbDocuments() {
    const container = document.getElementById('kb-documents-container');
    container.innerHTML = '<div>Loading runbooks...</div>';
    try {
      const res = await fetch('/api/kb/documents');
      const data = await res.json();
      if (!data.documents || data.documents.length === 0) {
        container.innerHTML = '<div style="color:var(--text-dim);">No documents found in documents/ directory.</div>';
        return;
      }
      container.innerHTML = data.documents.map(doc => `
        <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:var(--radius-md); padding:14px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-weight:700; color:var(--text-heading); font-family:var(--font-mono);">📄 ${escapeHtml(doc.filename)}</span>
            <span style="font-size:11px; color:var(--text-dim); font-weight:600;">${doc.lines_count} lines • ${(doc.size_bytes / 1024).toFixed(1)} KB</span>
          </div>
          <pre style="background:#ffffff; border:1px solid var(--border-color); padding:12px; border-radius:var(--radius-sm); font-size:12px; color:var(--text-body); max-height:150px; overflow-y:auto; font-family:var(--font-mono); line-height:1.5;">${escapeHtml(doc.preview)}...</pre>
        </div>
      `).join('');
    } catch (e) {
      container.innerHTML = `<div style="color:var(--text-red);">Error: ${escapeHtml(e.message)}</div>`;
    }
  }

  document.getElementById('btn-trigger-rebuild').addEventListener('click', function () {
    window.submitTerminalQuery('rebuild');
    closeModal(modalKb);
  });

  async function loadPresetsList() {
    const container = document.getElementById('presets-list-container');
    container.innerHTML = '<div>Loading presets...</div>';
    try {
      const res = await fetch('/api/presets');
      const presets = await res.json();
      container.innerHTML = presets.map(p => `
        <div style="background:#ffffff; border:1px solid var(--border-color); border-radius:var(--radius-md); padding:14px; display:flex; justify-content:space-between; align-items:center; cursor:pointer; box-shadow:0 1px 3px rgba(0,0,0,0.03); transition:all 0.15s;" onmouseover="this.style.borderColor='#3b82f6'; this.style.background='#eff6ff';" onmouseout="this.style.borderColor='var(--border-color)'; this.style.background='#ffffff';" onclick="window.submitTerminalQuery('${escapeJs(p.query)}'); document.getElementById('modal-presets').classList.remove('active');">
          <div style="display:flex; flex-direction:column; gap:4px; max-width:540px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="term-step-badge badge-${p.badge_color || 'cyan'}">${escapeHtml(p.badge)}</span>
              <strong style="color:var(--text-heading); font-size:13.5px; font-weight:700;">${escapeHtml(p.title)}</strong>
            </div>
            <div style="font-size:12px; color:var(--text-dim); font-style:italic;">"${escapeHtml(p.query)}"</div>
            <div style="font-size:11.5px; color:var(--text-muted); font-weight:500;">Expected: ${escapeHtml(p.expected)}</div>
          </div>
          <button class="btn-header primary" style="font-size:11.5px; padding:6px 12px;">Run ▶</button>
        </div>
      `).join('');
    } catch (e) {
      container.innerHTML = `<div style="color:var(--text-red);">Error: ${escapeHtml(e.message)}</div>`;
    }
  }

  async function loadSystemStatus() {
    const container = document.getElementById('system-hud-content');
    container.innerHTML = '<div>Fetching system metrics...</div>';
    try {
      const res = await fetch('/api/status');
      const s = await res.json();
      const keysHtml = Object.entries(s.keys_configured).map(([k, v]) => `
        <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f5f9;">
          <span style="font-family:var(--font-mono); color:var(--text-body); font-weight:500;">${k}</span>
          <span style="color:${v ? 'var(--text-green)' : 'var(--text-red)'}; font-weight:700;">${v ? 'LOADED' : 'NOT SET'}</span>
        </div>
      `).join('');

      container.innerHTML = `
        <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:var(--radius-md); padding:16px;">
          <h4 style="color:var(--text-heading); margin-bottom:10px; font-size:13.5px; font-weight:700;">System Configuration</h4>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:12.5px; color:var(--text-body);">
            <div>Engine Status: <strong style="color:var(--text-green);">ONLINE</strong></div>
            <div>Version: <strong style="color:var(--text-heading);">v${s.version}</strong></div>
            <div>Cost Priority Mode: <strong style="color:var(--text-heading);">${s.cost_priority_mode ? 'ENABLED (Free Tier Favoring)' : 'BALANCED'}</strong></div>
            <div>Default Provider: <strong style="color:var(--text-heading);">${s.default_provider}</strong></div>
            <div>FAISS Index Ready: <strong style="color:var(--text-green);">${s.faiss_ready ? 'YES' : 'NO'}</strong></div>
            <div>Runbooks Indexed: <strong style="color:var(--text-heading);">${s.runbooks_indexed}</strong></div>
            <div>Cache Status: <strong style="color:var(--text-green);">${s.cache_enabled ? 'ACTIVE' : 'DISABLED'}</strong></div>
            <div>Cached Records: <strong style="color:var(--text-cyan);">${s.cached_records_count} in DB</strong></div>
          </div>
        </div>

        <div style="background:#f8fafc; border:1px solid var(--border-color); border-radius:var(--radius-md); padding:16px;">
          <h4 style="color:var(--text-heading); margin-bottom:10px; font-size:13.5px; font-weight:700;">Configured LLM Provider Keys</h4>
          <div style="display:flex; flex-direction:column; gap:4px;">
            ${keysHtml}
          </div>
        </div>
      `;
    } catch (e) {
      container.innerHTML = `<div style="color:var(--text-red);">Error: ${escapeHtml(e.message)}</div>`;
    }
  }

  // --------------------------------------------------------------------------
  // Utility Functions
  // --------------------------------------------------------------------------
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function escapeJs(str) {
    if (!str) return '';
    return String(str).replace(/'/g, "\\'").replace(/"/g, '\\"');
  }

  function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
  }

  function formatMarkdownToHtml(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Format code blocks
    html = html.replace(/```([a-z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><button class="btn-copy-code" type="button">Copy</button><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code style="background:#f1f5f9; padding:1px 5px; border-radius:4px; font-family:var(--font-mono); color:#1d4ed8; border:1px solid #e2e8f0;">$1</code>');

    // Headings >> or ###
    html = html.replace(/^&gt;&gt;\s*(.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^#{1,3}\s*(.+)$/gm, '<h3>$1</h3>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong style="color:var(--text-heading); font-weight:700;">$1</strong>');

    // Bullet points
    html = html.replace(/^\*\s*(.+)$/gm, '<div style="margin-left:14px; margin-bottom:4px; color:var(--text-body);">• $1</div>');
    html = html.replace(/^-\s*(.+)$/gm, '<div style="margin-left:14px; margin-bottom:4px; color:var(--text-body);">• $1</div>');

    // Numbered steps
    html = html.replace(/^(\d+)\.\s*(.+)$/gm, '<div style="margin-left:14px; margin-bottom:4px; color:var(--text-body);"><strong style="color:#2563eb;">$1.</strong> $2</div>');

    // Newlines to breaks
    html = html.replace(/\n\n/g, '<div style="height:10px;"></div>');

    return html;
  }

  // Initial Boot
  connectWebSocket();

  // Fetch initial telemetry
  fetch('/api/cache/metrics')
    .then(r => r.json())
    .then(m => updateTelemetryMetrics(m))
    .catch(console.error);

})();

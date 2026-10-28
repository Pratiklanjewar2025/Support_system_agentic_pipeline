# -*- coding: utf-8 -*-
"""
AdaRAG-Ops AI Technical Support Dashboard Backend
FastAPI server providing WebSocket streaming and REST APIs for the multi-agent incident diagnosis pipeline.
"""

import os
import sys
import time
import json
import sqlite3
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

# Force UTF-8 encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Pipeline imports
from agents.state import AgentState
from agents.incident_response_cache import IncidentResponseCache
import rag.retriever as retriever

app = FastAPI(
    title="AdaRAG-Ops Technical Support Dashboard",
    description="Real-time multi-agent autonomous SRE support dashboard with interactive terminal",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global singleton graph
_GRAPH_INSTANCE = None
_GRAPH_LOCK = threading.Lock()


def get_compiled_graph():
    global _GRAPH_INSTANCE
    with _GRAPH_LOCK:
        if _GRAPH_INSTANCE is None:
            from agents.orchestrator import build_graph
            _GRAPH_INSTANCE = build_graph()
        return _GRAPH_INSTANCE


def get_cache():
    return IncidentResponseCache.get_instance()


@app.on_event("startup")
async def startup_event():
    get_cache().ensure_presets_seeded()


# -----------------------------------------------------------------------------
# Curated Preset Incidents for Instant Testing
# -----------------------------------------------------------------------------
PRESET_INCIDENTS = [
    {
        "id": "cache_hit_checkout",
        "title": "Restart Checkout Service (Instant Cache Hit)",
        "query": "How do I restart the checkout service?",
        "category": "deployment",
        "expected": "Instant 0-Cost Cache Hit (<15ms)",
        "badge": "CACHE HIT",
        "badge_color": "emerald"
    },
    {
        "id": "postgres_pool",
        "title": "PostgreSQL Memory & Connection Spike",
        "query": "PostgreSQL database memory is getting exhausted with spikes in active connections and max_connections exceeded.",
        "category": "database",
        "expected": "Full Multi-Agent Diagnostic RCA & PgBouncer Fix",
        "badge": "MULTI-AGENT",
        "badge_color": "cyan"
    },
    {
        "id": "k8s_oom",
        "title": "Kubernetes Pod CrashLoopBackOff (OOMKilled)",
        "query": "Kubernetes API gateway pod is in CrashLoopBackOff with OOMKilled exit code 137.",
        "category": "deployment",
        "expected": "OOM Diagnostic Analysis & Memory Limits Remediations",
        "badge": "CACHE / RCA",
        "badge_color": "violet"
    },
    {
        "id": "checkout_500",
        "title": "Checkout API 500 Error Spike",
        "query": "Checkout API returning 500 Internal Server Error with database connection timeout and latency spikes.",
        "category": "application",
        "expected": "Knowledge Retrieval & Circuit Breaker Remediation",
        "badge": "RAG SEARCH",
        "badge_color": "amber"
    },
    {
        "id": "payments_cpu",
        "title": "Payments Service 100% CPU Saturation",
        "query": "Payments service experiencing sustained 100% CPU utilization with thread starvation and blocked worker pools.",
        "category": "cpu",
        "expected": "Thread Contention RCA & Worker Scaling SOP",
        "badge": "L2 DIAGNOSIS",
        "badge_color": "cyan"
    },
    {
        "id": "rollback_sop",
        "title": "Production Deployment Rollback SOP",
        "query": "What is the standard operating procedure to rollback a failed canary deployment in kubernetes safely?",
        "category": "deployment",
        "expected": "SOP Runbook Retrieval & Safe Execution Steps",
        "badge": "RUNBOOK SOP",
        "badge_color": "emerald"
    }
]


# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Dashboard index.html not found.</h1>", status_code=404)
    return FileResponse(str(index_file))


@app.get("/api/status")
async def get_system_status():
    """Returns system status, active LLM API keys, FAISS index state, and cache status."""
    keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "COHERE_API_KEY": bool(os.getenv("COHERE_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
    
    # Check FAISS index
    index_file = BASE_DIR / "faiss_index" / "index.faiss"
    faiss_ready = index_file.exists()
    
    # Document count
    docs_dir = BASE_DIR / "documents"
    doc_count = len(list(docs_dir.glob("*.md"))) if docs_dir.exists() else 0

    cache = get_cache()
    cache_metrics = cache.get_metrics()

    return {
        "status": "online",
        "version": "2.0.0",
        "cost_priority_mode": os.getenv("COST_PRIORITY_MODE", "false").lower() in ("true", "1", "yes"),
        "default_provider": os.getenv("LLM_PROVIDER", "auto"),
        "default_model": os.getenv("LLM_MODEL_NAME", "adaptive"),
        "keys_configured": keys,
        "faiss_ready": faiss_ready,
        "runbooks_indexed": doc_count,
        "cache_enabled": cache.enabled,
        "cached_records_count": cache_metrics.get("total_records_in_db", 0),
        "total_lookups": cache_metrics.get("total_lookups", 0),
        "hit_rate_percent": cache_metrics.get("hit_rate_percent", 0.0),
        "avoided_cost_usd": cache_metrics.get("avoided_llm_cost_usd", 0.0),
    }


@app.get("/api/presets")
async def get_presets():
    return PRESET_INCIDENTS


@app.get("/api/cache/metrics")
async def get_cache_telemetry():
    cache = get_cache()
    return cache.get_metrics()


@app.get("/api/cache/records")
async def get_cached_records(limit: int = 50):
    cache = get_cache()
    records = []
    try:
        with sqlite3.connect(cache.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT cache_id, query, affected_service, severity, incident_category,
                       hit_count, created_at, root_cause, resolution_status, model_id, is_invalidated
                FROM incident_response_cache
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            rows = cur.fetchall()
            for r in rows:
                records.append({
                    "cache_id": r[0],
                    "query": r[1],
                    "affected_service": r[2],
                    "severity": r[3],
                    "incident_category": r[4],
                    "hit_count": r[5],
                    "created_at": r[6],
                    "root_cause": (r[7][:160] + "...") if r[7] and len(r[7]) > 160 else r[7],
                    "resolution_status": r[8],
                    "model_id": r[9],
                    "is_invalidated": bool(r[10])
                })
    except Exception as e:
        return JSONResponse({"error": str(e), "records": []}, status_code=500)
    return {"records": records, "total": len(records)}


@app.post("/api/cache/clear")
async def clear_cache_records():
    cache = get_cache()
    cleared = cache.clear_cache()
    return {"status": "success", "cleared_count": cleared}


@app.post("/api/cache/toggle")
async def toggle_cache_state():
    cache = get_cache()
    cache.enabled = not cache.enabled
    return {"status": "success", "cache_enabled": cache.enabled}


@app.post("/api/cache/invalidate/{cache_id}")
async def invalidate_cache_entry(cache_id: str):
    cache = get_cache()
    success = cache.invalidate_record(cache_id, reason="dashboard_manual_invalidation")
    return {"status": "success" if success else "failed", "cache_id": cache_id}


class CacheAddRequest(BaseModel):
    query: str
    root_cause: str
    resolution_steps: str
    category: Optional[str] = "general"
    severity: Optional[str] = "L1"
    affected_service: Optional[str] = None


@app.post("/api/cache/add")
async def add_cache_entry(req: CacheAddRequest):
    cache = get_cache()
    cache_id = cache.store_query(
        query=req.query,
        root_cause=req.root_cause,
        resolution_steps=req.resolution_steps,
        category=req.category or "general",
        severity=req.severity or "L1",
        affected_service=req.affected_service,
    )
    if not cache_id:
        raise HTTPException(status_code=400, detail="Failed to persist query to incident cache.")
    return {"status": "success", "cache_id": cache_id}


@app.get("/api/kb/documents")
async def list_kb_documents():
    docs_dir = BASE_DIR / "documents"
    docs = []
    if docs_dir.exists():
        for f in sorted(docs_dir.glob("*.md")):
            content = f.read_text(encoding="utf-8", errors="replace")
            docs.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "preview": content[:400],
                "full_content": content,
                "lines_count": len(content.splitlines())
            })
    return {"documents": docs, "count": len(docs)}


@app.post("/api/kb/rebuild")
async def rebuild_knowledge_base():
    try:
        retriever.build_index()
        return {"status": "success", "message": "FAISS vector index rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Streaming Pipeline Execution Generator
# -----------------------------------------------------------------------------
STEP_META = {
    "cache_lookup": {
        "label": "Incident Cache Guard",
        "desc": "Checking persistent SQLite cache for verified diagnosis...",
        "icon": "shield-check"
    },
    "classify": {
        "label": "Ticket Classifier",
        "desc": "Extracting category, severity (L1-L3), and problem summary...",
        "icon": "tag"
    },
    "knowledge_search": {
        "label": "Knowledge Retrieval (RAG)",
        "desc": "Semantic search over runbooks and incident documentation...",
        "icon": "book-open"
    },
    "evidence_assessment": {
        "label": "Evidence Auditor",
        "desc": "Assessing lexical coverage, retrieval confidence & missing telemetry...",
        "icon": "search"
    },
    "router": {
        "label": "Adaptive Model Router",
        "desc": "Pareto utility evaluation & heterogeneous model tier selection...",
        "icon": "cpu"
    },
    "diagnose": {
        "label": "Diagnostic RCA Agent",
        "desc": "Evidence-grounded root cause analysis and hypothesis deduction...",
        "icon": "activity"
    },
    "resolve": {
        "label": "Resolution Engine",
        "desc": "Synthesizing actionable advisory commands and remediation steps...",
        "icon": "tool"
    },
    "verify": {
        "label": "Verification Agent",
        "desc": "Claim-level safety verification and calibrated confidence assessment...",
        "icon": "check-circle"
    },
    "retrieve_more": {
        "label": "Active Re-Retrieval",
        "desc": "Deepening retrieval window to gather additional incident context...",
        "icon": "refresh-cw"
    },
    "switch_model": {
        "label": "Model Tier Escalation",
        "desc": "Upgrading model tier to high-capacity reasoning model...",
        "icon": "arrow-up-circle"
    },
    "abstain": {
        "label": "Diagnostic Abstention",
        "desc": "Declining automated remediation to protect production safety...",
        "icon": "slash"
    },
    "escalate": {
        "label": "Human SRE Escalation",
        "desc": "Routing ticket to human on-call engineer with complete audit trail...",
        "icon": "alert-triangle"
    },
    "close": {
        "label": "Ticket Resolution",
        "desc": "Packaging verified RCA, updating telemetry and persisting to cache...",
        "icon": "flag"
    }
}


def clean_markdown_for_terminal(text: str) -> str:
    """Format markdown for clean reading."""
    if not text:
        return ""
    import re
    cleaned = text
    cleaned = re.sub(r'```[a-z]*\n?', '', cleaned)
    cleaned = re.sub(r'^#{1,6}\s*(.+)$', r'>> \1', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


async def execute_pipeline_streaming(query: str, websocket: WebSocket):
    """
    Executes LangGraph multi-agent pipeline and streams step-by-step
    events over WebSocket in real time.
    """
    graph = get_compiled_graph()

    initial_state: dict = {
        "user_query": query,
        "category": "",
        "severity": "",
        "summary": "",
        "kb_results": [],
        "kb_context": "",
        "root_cause": "",
        "confidence": "",
        "resolution_steps": "",
        "resolution_status": "",
        "verified": False,
        "verification_notes": "",
        "escalated": False,
        "escalation_reason": "",
        "final_answer": "",
        "cache_hit": False,
        "cache_match_type": "miss",
        "cache_similarity": 0.0,
        "cache_record_id": None,
        "cache_lookup_latency": 0.0,
        "start_time": time.time(),
    }

    loop = asyncio.get_running_loop()
    event_queue = asyncio.Queue()

    def run_graph_thread():
        try:
            for event in graph.stream(initial_state):
                for node_name, state in event.items():
                    loop.call_soon_threadsafe(event_queue.put_nowait, ("node", node_name, state))
            loop.call_soon_threadsafe(event_queue.put_nowait, ("complete", None, None))
        except Exception as exc:
            loop.call_soon_threadsafe(event_queue.put_nowait, ("error", str(exc), None))

    threading.Thread(target=run_graph_thread, daemon=True).start()

    t_start = time.time()
    final_state = None
    step_index = 0

    # Initial notification
    await websocket.send_json({
        "type": "pipeline_start",
        "query": query,
        "timestamp": t_start
    })

    while True:
        msg_type, node_name, state = await event_queue.get()
        if msg_type == "complete":
            break
        elif msg_type == "error":
            await websocket.send_json({
                "type": "pipeline_error",
                "error": node_name,
                "timestamp": time.time()
            })
            return

        step_index += 1
        final_state = state
        node_meta = STEP_META.get(node_name, {"label": node_name, "desc": "Processing...", "icon": "cpu"})
        step_elapsed = round(time.time() - t_start, 3)

        # Extract structured details for this specific node
        node_data = {
            "node_name": node_name,
            "step_index": step_index,
            "label": node_meta["label"],
            "desc": node_meta["desc"],
            "icon": node_meta["icon"],
            "elapsed_sec": step_elapsed,
        }

        if node_name == "cache_lookup":
            hit = state.get("cache_hit", False)
            match_type = state.get("cache_match_type", "miss")
            sim = state.get("cache_similarity", 0.0)
            lat = state.get("cache_lookup_latency", 0.0)
            rec_id = state.get("cache_record_id", None)
            node_data.update({
                "hit": hit,
                "match_type": match_type,
                "similarity": sim,
                "lookup_latency": lat,
                "record_id": rec_id,
                "cost_saved": "$0.00500" if hit else "$0.00"
            })

        elif node_name == "classify":
            node_data.update({
                "category": state.get("category", "general"),
                "severity": state.get("severity", "L2"),
                "summary": state.get("summary", "")
            })

        elif node_name == "knowledge_search":
            results = state.get("kb_results", [])
            sources = list({r.get("source", "unknown") for r in results})
            snippets = [
                {"source": r.get("source"), "score": round(r.get("score", 0.0), 3), "snippet": r.get("text", "")[:140]}
                for r in results[:4]
            ]
            node_data.update({
                "chunks_count": len(results),
                "sources": sources,
                "snippets": snippets
            })

        elif node_name == "evidence_assessment":
            rm = state.get("routing_metadata") or {}
            rf = rm.get("retrieval_features", {})
            node_data.update({
                "retrieval_confidence": round(rf.get("retrieval_confidence", 0.0) * 100, 1),
                "top_1_similarity": round(rf.get("top_1_similarity", 0.0), 4),
                "missing_evidence": state.get("missing_evidence", [])
            })

        elif node_name == "router":
            rm = state.get("routing_metadata") or {}
            qf = rm.get("query_features", {})
            rf = rm.get("retrieval_features", {})
            node_data.update({
                "chosen_model": rm.get("chosen_model", state.get("selected_model", "Default")),
                "provider": (rm.get("provider") or state.get("selected_provider", "auto")).upper(),
                "chosen_tier": rm.get("chosen_tier", "tier1_fast"),
                "complexity_score": qf.get("complexity_score", 0.0),
                "retrieval_confidence": round(rf.get("retrieval_confidence", 0.0) * 100, 1),
                "recommended_action": state.get("recommended_action", "ANSWER")
            })

        elif node_name == "diagnose":
            node_data.update({
                "confidence": state.get("confidence", "medium"),
                "root_cause": state.get("root_cause", ""),
                "affected_component": state.get("affected_component", "system")
            })

        elif node_name == "resolve":
            node_data.update({
                "resolution_status": state.get("resolution_status", "resolved"),
                "steps": state.get("resolution_steps", "")
            })

        elif node_name == "verify":
            calib = state.get("calibrated_confidence", 0.85) or 0.85
            node_data.update({
                "verified": state.get("verified", False),
                "verification_notes": state.get("verification_notes", ""),
                "calibrated_confidence": round(calib * 100, 1),
                "verification_status": state.get("verification_status", "verified")
            })

        elif node_name in ("close", "escalate", "abstain"):
            node_data.update({
                "escalated": state.get("escalated", False),
                "escalation_reason": state.get("escalation_reason", ""),
                "final_answer": state.get("final_answer", "")
            })

        # Send step event to frontend
        await websocket.send_json({
            "type": "step_update",
            "node": node_data,
            "timestamp": time.time()
        })
        # Small non-blocking yield to allow front-end animation
        await asyncio.sleep(0.04)

    total_elapsed = round(time.time() - t_start, 2)
    final_answer = (final_state or {}).get("final_answer", "No answer generated.")
    cache_hit = (final_state or {}).get("cache_hit", False)
    escalated = (final_state or {}).get("escalated", False)
    verified = (final_state or {}).get("verified", False)
    category = (final_state or {}).get("category", "General")
    severity = (final_state or {}).get("severity", "L2")
    model_used = (final_state or {}).get("selected_model", "Cache")
    
    # Send final pipeline completion payload
    cache_metrics = get_cache().get_metrics()

    await websocket.send_json({
        "type": "pipeline_complete",
        "elapsed_sec": total_elapsed,
        "cache_hit": cache_hit,
        "cache_match_type": (final_state or {}).get("cache_match_type", "miss"),
        "escalated": escalated,
        "verified": verified,
        "category": category,
        "severity": severity,
        "model_used": model_used,
        "root_cause": (final_state or {}).get("root_cause", ""),
        "resolution_steps": (final_state or {}).get("resolution_steps", ""),
        "final_answer": final_answer,
        "cache_metrics": cache_metrics,
        "timestamp": time.time()
    })


# -----------------------------------------------------------------------------
# WebSocket Handler (Interactive Terminal + Commands)
# -----------------------------------------------------------------------------
@app.websocket("/ws/terminal")
async def terminal_websocket(websocket: WebSocket):
    await websocket.accept()
    cache = get_cache()

    # Initial Welcome message
    await websocket.send_json({
        "type": "terminal_init",
        "message": "Connected to AdaRAG-Ops Autonomous Technical Support Engine.",
        "cache_records": cache.get_metrics().get("total_records_in_db", 0),
        "cache_enabled": cache.enabled
    })

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except Exception:
                payload = {"command": data.strip()}

            raw_input = payload.get("command", "").strip()
            if not raw_input:
                continue

            cmd_lower = raw_input.lower()

            # Handle terminal CLI commands
            if cmd_lower in ("help", "?"):
                help_text = (
                    "Available Terminal Commands:\n"
                    "  help           - Display this command reference\n"
                    "  cache          - Inspect Incident Response Cache metrics and saved records\n"
                    "  cache-clear    - Purge all persistent cache records from SQLite\n"
                    "  cache-toggle   - Enable / Disable incident response cache lookup\n"
                    "  rebuild        - Rebuild the FAISS knowledge base vector index\n"
                    "  status         - Display LLM provider API keys and active models\n"
                    "  presets        - List pre-built operational incident scenarios\n"
                    "  clear          - Clear the terminal screen\n"
                    "  exit / quit    - End terminal session\n\n"
                    "Or simply type any technical issue or question (e.g. 'Postgres memory exhaustion', 'K8s OOMKilled')."
                )
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": help_text,
                    "level": "info"
                })

            elif cmd_lower == "cache":
                m = cache.get_metrics()
                # Also fetch recent records
                records_preview = []
                try:
                    with sqlite3.connect(cache.db_path) as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT query, affected_service, hit_count FROM incident_response_cache WHERE is_invalidated=0 ORDER BY created_at DESC LIMIT 5")
                        records_preview = cur.fetchall()
                except Exception:
                    pass

                out_lines = [
                    "=" * 60,
                    "  INCIDENT RESPONSE CACHE TELEMETRY & OBSERVABILITY",
                    "=" * 60,
                    f"  Cache Enabled          : {'YES' if m['cache_enabled'] else 'NO'}",
                    f"  Semantic Cosine Search : {'YES' if m['semantic_enabled'] else 'NO'} (Threshold: {m['similarity_threshold']:.2f})",
                    f"  Records in Database    : {m['total_records_in_db']} (TTL: {int(m['ttl_seconds'])}s)",
                    f"  Total Lookups          : {m['total_lookups']}",
                    f"  Exact Cache Hits       : {m['exact_hits']}",
                    f"  Semantic Cache Hits    : {m['semantic_hits']}",
                    f"  Cache Misses           : {m['misses']}",
                    f"  Overall Hit Rate       : {m['hit_rate_percent']}%",
                    f"  Avoided Model Calls    : {m['avoided_model_calls']}",
                    f"  Avoided LLM Cost (Est) : ${m['avoided_llm_cost_usd']:.5f}",
                    f"  Avg Lookup Latency     : {m['avg_lookup_latency_sec']:.5f}s",
                    "-" * 60,
                    "  Recent Verified Cached Queries:"
                ]
                if records_preview:
                    for q, svc, hits in records_preview:
                        tag = f"[{svc}] " if svc else ""
                        out_lines.append(f"    * {tag}{q} (hits: {hits})")
                else:
                    out_lines.append("    (No records in cache yet)")
                out_lines.append("=" * 60)

                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": "\n".join(out_lines),
                    "level": "success",
                    "cache_metrics": m
                })

            elif cmd_lower == "cache-clear":
                cleared = cache.clear_cache()
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": f"[OK] Cache cleared: {cleared} records purged from persistent storage.",
                    "level": "warning",
                    "cache_metrics": cache.get_metrics()
                })

            elif cmd_lower == "cache-toggle":
                cache.enabled = not cache.enabled
                status_str = "ENABLED" if cache.enabled else "DISABLED"
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": f"[OK] Incident Response Cache is now {status_str}.",
                    "level": "info",
                    "cache_metrics": cache.get_metrics()
                })

            elif cmd_lower == "rebuild":
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": "[*] Rebuilding FAISS vector index from documents...",
                    "level": "info"
                })
                try:
                    retriever.build_index()
                    await websocket.send_json({
                        "type": "terminal_command_result",
                        "command": "rebuild-done",
                        "output": "[OK] Knowledge base index rebuilt successfully and ready.",
                        "level": "success"
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "terminal_command_result",
                        "command": "rebuild-error",
                        "output": f"[ERROR] Failed to rebuild index: {e}",
                        "level": "error"
                    })

            elif cmd_lower == "clear":
                await websocket.send_json({
                    "type": "terminal_clear"
                })

            elif cmd_lower == "status":
                status_dict = await get_system_status()
                keys_list = [k for k, v in status_dict["keys_configured"].items() if v]
                out = [
                    "=" * 60,
                    "  SYSTEM & PROVIDER STATUS",
                    "=" * 60,
                    f"  Status              : {status_dict['status'].upper()}",
                    f"  Pipeline Engine     : AdaRAG-Ops v{status_dict['version']}",
                    f"  Configured API Keys : {', '.join(keys_list) if keys_list else 'None'}",
                    f"  FAISS Vector Index  : {'Ready' if status_dict['faiss_ready'] else 'Missing'}",
                    f"  Runbooks Indexed    : {status_dict['runbooks_indexed']}",
                    f"  Cost Priority Mode  : {status_dict['cost_priority_mode']}",
                    f"  Default Provider    : {status_dict['default_provider']}",
                    f"  Cache Status        : {'ACTIVE (' + str(status_dict['cached_records_count']) + ' records)' if status_dict['cache_enabled'] else 'DISABLED'}",
                    "=" * 60
                ]
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": "\n".join(out),
                    "level": "info"
                })

            elif cmd_lower in ("exit", "quit"):
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": "Session preserved. Close browser tab to disconnect.",
                    "level": "info"
                })

            elif cmd_lower == "presets":
                p_lines = ["Available Incident Presets:"]
                for p in PRESET_INCIDENTS:
                    p_lines.append(f"  [{p['badge']}] {p['title']}")
                    p_lines.append(f"      Query: \"{p['query']}\"")
                await websocket.send_json({
                    "type": "terminal_command_result",
                    "command": raw_input,
                    "output": "\n".join(p_lines),
                    "level": "info"
                })

            else:
                # Normal operational query: execute the full multi-agent pipeline
                await execute_pipeline_streaming(raw_input, websocket)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] {e}")

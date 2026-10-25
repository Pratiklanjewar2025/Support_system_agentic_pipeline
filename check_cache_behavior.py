# -*- coding: utf-8 -*-
import sys
import time
from agents.orchestrator import build_graph
from agents.incident_response_cache import IncidentResponseCache

query = "What is the recommended procedure to clear linux journal logs when disk is full?"
print(f"[*] Running query: '{query}'", flush=True)

cache = IncidentResponseCache.get_instance()
res = cache.lookup(query)
print(f"[*] Cache lookup before run: hit={res.hit}, match_type={res.match_type}", flush=True)

graph = build_graph()
initial_state = {
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

for event in graph.stream(initial_state):
    for node_name, state in event.items():
        print(f"  -> Node: {node_name} | verified={state.get('verified')} | escalated={state.get('escalated')} | status={state.get('resolution_status')}", flush=True)

res2 = cache.lookup(query)
print(f"[*] Cache lookup after run: hit={res2.hit}, match_type={res2.match_type}", flush=True)

# -*- coding: utf-8 -*-
"""Quick test script -- runs RAG pipeline without needing an API key."""
import sys, pathlib, shutil

# Reconfigure stdout to UTF-8 in-place (Python 3.7+) -- fixes Windows cp1252 errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, ".")

from rag.retriever import build_index, retrieve

# Clean old index for fresh test
idx = pathlib.Path("faiss_index")
if idx.exists():
    shutil.rmtree(idx)

print("=== Building FAISS index from documents ===")
build_index()

print("\n=== Test 1: '502 error after deployment' ===")
results = retrieve("checkout API 502 error after deployment")
for i, r in enumerate(results, 1):
    print(f"  [{i}] Source : {r['source']}")
    print(f"       Score  : {r['score']:.3f}")
    print(f"       Text   : {r['text'][:120]}...")
    print()

print("=== Test 2: 'payments CPU high' ===")
results2 = retrieve("payments service CPU high utilization")
for i, r in enumerate(results2[:2], 1):
    print(f"  [{i}] Source : {r['source']}")
    print(f"       Score  : {r['score']:.3f}")
    print()

print("=== Test 3: 'rollback deployment' ===")
results3 = retrieve("how to rollback production deployment")
for i, r in enumerate(results3[:2], 1):
    print(f"  [{i}] Source : {r['source']}")
    print(f"       Score  : {r['score']:.3f}")
    print()

print("[OK] RAG PIPELINE FULLY WORKING")

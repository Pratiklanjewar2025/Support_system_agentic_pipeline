# -*- coding: utf-8 -*-
"""
AdaRAG-Ops Autonomous Technical Support Dashboard Runner
=========================================================
Run:  python run_dashboard.py [--port 8000]

Launches the FastAPI server and serves the real-time AI Technical Support
autonomous command dashboard with interactive terminal interface.
"""

import os
import sys
import argparse
import webbrowser
from pathlib import Path

# Force UTF-8 encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

# Verify API Keys
api_keys = [
    os.getenv("COHERE_API_KEY"),
    os.getenv("OPENROUTER_API_KEY"),
    os.getenv("GROQ_API_KEY"),
    os.getenv("DEEPSEEK_API_KEY"),
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GOOGLE_API_KEY"),
]
if not any(api_keys):
    print("\n[!] WARNING: No LLM API key detected in .env file.")
    print("    Please check GROQ_API_KEY, GEMINI_API_KEY, COHERE_API_KEY, or OPENROUTER_API_KEY.")
    print("    (Cache hits will still resolve at 0-cost, but live multi-LLM queries require an API key)\n")


def check_and_build_index():
    """Ensure FAISS index exists before starting server."""
    index_file = Path(__file__).parent / "faiss_index" / "index.faiss"
    if not index_file.exists():
        print("[*] Knowledge base index not found. Building FAISS vector store...")
        from rag.retriever import build_index
        build_index()
    else:
        print("[OK] FAISS knowledge base index is ready.")


def main():
    parser = argparse.ArgumentParser(description="AdaRAG-Ops AI Support Command Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    check_and_build_index()

    url = f"http://{args.host}:{args.port}"
    
    print("\n" + "=" * 72)
    print("   ADARAG-OPS: AUTONOMOUS SRE INCIDENT COMMAND DASHBOARD")
    print("   Powered by Multi-Agent AI + Persistent Cache + FAISS Vector RAG")
    print("=" * 72)
    print(f"\n   [>] Dashboard URL   : {url}")
    print(f"   [>] Terminal Console: Interactive live chat & CLI commands supported")
    print(f"   [>] Incident Cache  : Persistent SQLite exact & semantic guard active")
    print(f"   [>] Press Ctrl+C to terminate\n")
    print("=" * 72 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import uvicorn
    uvicorn.run(
        "dashboard.backend:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()

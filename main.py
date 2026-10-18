# -*- coding: utf-8 -*-
"""
AI Technical Support Automation Platform -- Terminal Interface
==============================================================
Run:  python main.py

Type your support problem and watch the multi-agent pipeline diagnose and resolve it.
Type 'quit' or 'exit' to stop. Type 'rebuild' to re-index the knowledge base.
"""

import os
import sys
import time
import re
from pathlib import Path

# Reconfigure stdout/stderr to UTF-8 in-place (Python 3.7+) -- fixes Windows cp1252 errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

# -- Load environment variables ----------------------------------------------
load_dotenv()
api_keys = [os.getenv("COHERE_API_KEY"), os.getenv("OPENROUTER_API_KEY"), os.getenv("GROQ_API_KEY"), os.getenv("DEEPSEEK_API_KEY")]
if not any(api_keys):
    print("\n[ERROR] No LLM API key found.")
    print("    Please add COHERE_API_KEY, OPENROUTER_API_KEY, or GROQ_API_KEY to your .env file.")
    sys.exit(1)

# -- Colorama for coloured terminal output -----------------------------------
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def cyan(s):    return Fore.CYAN    + str(s) + Style.RESET_ALL
    def green(s):   return Fore.GREEN   + str(s) + Style.RESET_ALL
    def yellow(s):  return Fore.YELLOW  + str(s) + Style.RESET_ALL
    def red(s):     return Fore.RED     + str(s) + Style.RESET_ALL
    def bold(s):    return Style.BRIGHT + str(s) + Style.RESET_ALL
    def magenta(s): return Fore.MAGENTA + str(s) + Style.RESET_ALL
    def blue(s):    return Fore.BLUE    + str(s) + Style.RESET_ALL
    def white(s):   return Fore.WHITE   + str(s) + Style.RESET_ALL
except ImportError:
    def cyan(s):    return str(s)
    def green(s):   return str(s)
    def yellow(s):  return str(s)
    def red(s):     return str(s)
    def bold(s):    return str(s)
    def magenta(s): return str(s)
    def blue(s):    return str(s)
    def white(s):   return str(s)


BANNER = """
  +=============================================================+
  |         AI Technical Support Assistant                      |
  |         Powered by Multi-Agent AI + Knowledge Base          |
  +=============================================================+
"""

SEVERITY_LABEL  = {"L1": "Low",  "L2": "Medium", "L3": "High"}
SEVERITY_COLORS = {"L1": green,  "L2": yellow,    "L3": red}
CONFIDENCE_COLORS = {"high": green, "medium": yellow, "low": red}

STEP_MESSAGES = {
    "classify":         ("Step 1 of 6", "Understanding your issue..."),
    "knowledge_search": ("Step 2 of 6", "Searching the knowledge base..."),
    "router":           ("Step 3 of 6", "Adaptive Model Router evaluating query & retrieval confidence..."),
    "diagnose":         ("Step 4 of 6", "Diagnosing the root cause..."),
    "resolve":          ("Step 5 of 6", "Generating a fix..."),
    "verify":           ("Step 6 of 6", "Verifying the solution..."),
    "model_escalate":   ("Escalation",  "Escalating query to high-capacity reasoning model..."),
    "escalate":         ("Escalation",  "Escalating to a human engineer..."),
    "close":            ("Resolution",  "Wrapping up..."),
}


def divider(char="-", width=66):
    print(cyan(char * width))


def clean_markdown(text: str) -> str:
    """Strip markdown so the text reads as plain English."""
    text = re.sub(r'```[a-z]*\n?', '', text)
    text = re.sub(r'^#{1,6}\s*(.+)$', r'>> \1', text, flags=re.MULTILINE)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def wrap_text(text: str, width: int = 62, indent: str = "    ") -> str:
    """Word-wrap a long paragraph."""
    words = text.split()
    lines, current, length = [], [], 0
    for word in words:
        if length + len(word) + 1 > width:
            lines.append(indent + " ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(indent + " ".join(current))
    return "\n".join(lines)


def print_banner():
    print(cyan(BANNER))


def print_step(node_name: str):
    label, msg = STEP_MESSAGES.get(node_name, ("...", "Working..."))
    print(f"\n  {cyan('>>')} {bold(label)}  --  {msg}")


def print_classify_result(state: dict):
    sev     = state.get("severity", "?")
    col     = SEVERITY_COLORS.get(sev, yellow)
    sev_lbl = SEVERITY_LABEL.get(sev, sev)
    cat     = state.get("category", "?").capitalize()
    summary = state.get("summary", "")
    print(f"       Category : {magenta(cat)}")
    print(f"       Priority : {col(sev_lbl)}")
    if summary:
        print(f"       Summary  : {summary}")


def print_kb_result(state: dict):
    results = state.get("kb_results", [])
    if not results:
        print(f"       {yellow('No relevant documents found - using general knowledge.')}")
        return
    sources = list({r["source"] for r in results[:4]})
    print(f"       Found {len(results)} relevant document(s):")
    for src in sources:
        print(f"         - {green(src)}")


def print_router_result(state: dict):
    rm = state.get("routing_metadata") or {}
    tier = rm.get("chosen_tier", "tier1_fast")
    model_name = rm.get("chosen_model", state.get("selected_model", "Default"))
    rf = rm.get("retrieval_features", {})
    conf = rf.get("retrieval_confidence", 0.0)
    qf = rm.get("query_features", {})
    comp = qf.get("complexity_score", 0.0)
    print(f"       Routed Model : {cyan(model_name)} [{magenta(tier)}]")
    print(f"       Complexity   : {yellow(str(comp))} | Retrieval Confidence : {green(f'{conf*100:.1f}%')}")


def print_diagnosis_result(state: dict):
    conf  = state.get("confidence", "?")
    col   = CONFIDENCE_COLORS.get(conf, yellow)
    cause = state.get("root_cause", "Unknown")
    print(f"       Confidence : {col(conf.capitalize())}")
    print()
    for line in cause.split("\n"):
        if line.strip():
            print(wrap_text(line, indent="       "))


def print_resolution_result(state: dict):
    status = state.get("resolution_status", "")
    col    = green if status == "resolved" else yellow
    label  = "Solution ready" if status == "resolved" else "Needs human review"
    print(f"       Status : {col(label)}")


def print_verification_result(state: dict):
    verified = state.get("verified", False)
    if verified:
        print(f"       {green('Solution verified as safe and complete.')}")
    else:
        notes = state.get("verification_notes", "")
        print(f"       {yellow('Flagged for human review.')}")
        if notes:
            print(f"       Reason : {notes}")


def print_final_answer(state: dict, elapsed: float):
    escalated = state.get("escalated", False)
    answer    = clean_markdown(state.get("final_answer", "No answer generated."))

    print()
    divider("=")
    if escalated:
        print(f"  {red(bold('  ESCALATED  --  A human engineer needs to review this'))}")
    else:
        print(f"  {green(bold('  RESOLVED  --  Here is your answer'))}")
    divider("=")
    print()

    for line in answer.split("\n"):
        stripped = line.strip()
        if not stripped:
            print()
        elif stripped.startswith(">>"):
            heading = stripped.lstrip("> ").strip()
            print(f"\n  {bold(heading)}")
            print(f"  {'-' * len(heading)}")
        elif re.match(r'^\d+\.', stripped):
            num, rest = stripped.split(".", 1)
            print(f"    {bold(cyan(num + '.'))} {rest.strip()}")
        elif stripped.startswith(("-", "*", "+")):
            content = stripped.lstrip("-*+ ").strip()
            print(f"      {cyan('*')} {content}")
        else:
            print(wrap_text(stripped, indent="    "))

    print()
    divider("=")
    print(f"  {cyan('Time taken:')} {elapsed:.1f} seconds")
    divider("=")
    print()


def run_pipeline(query: str):
    """Run the full agent pipeline for a user query and print each step."""
    from agents.orchestrator import build_graph

    graph = build_graph()

    initial_state: dict = {
        "user_query":         query,
        "category":           "",
        "severity":           "",
        "summary":            "",
        "kb_results":         [],
        "kb_context":         "",
        "root_cause":         "",
        "confidence":         "",
        "resolution_steps":   "",
        "resolution_status":  "",
        "verified":           False,
        "verification_notes": "",
        "escalated":          False,
        "escalation_reason":  "",
        "final_answer":       "",
    }

    print()
    divider()
    print(f"  {bold('Your question:')}  {yellow(query)}")
    divider()
    print(f"  {cyan('Please wait while the AI works on your issue...')}")

    final_state = None
    t_start     = time.time()

    for event in graph.stream(initial_state):
        for node_name, state in event.items():
            if node_name not in STEP_MESSAGES:
                continue
            print_step(node_name)
            if node_name == "classify":
                print_classify_result(state)
            elif node_name == "knowledge_search":
                print_kb_result(state)
            elif node_name == "router":
                print_router_result(state)
            elif node_name == "diagnose":
                print_diagnosis_result(state)
            elif node_name == "resolve":
                print_resolution_result(state)
            elif node_name == "verify":
                print_verification_result(state)
            final_state = state

    elapsed = time.time() - t_start
    if final_state:
        print_final_answer(final_state, elapsed)


def rebuild_index():
    """Force rebuild the FAISS index."""
    index_path = Path("faiss_index")
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)
        print(yellow("  Old index removed."))
    from rag.retriever import build_index
    build_index()


def main():
    print_banner()

    index_file = Path("faiss_index") / "index.faiss"
    if not index_file.exists():
        print(cyan("  Building knowledge base for the first time, please wait...\n"))
        from rag.retriever import build_index
        build_index()
    else:
        print(green("  Knowledge base is ready.\n"))

    divider()
    print(f"  {bold('How to use:')}")
    print(f"    * Describe your technical problem and press Enter")
    print(f"    * Type  {cyan('rebuild')}  to refresh the knowledge base")
    print(f"    * Type  {cyan('exit')}     to quit")
    divider()

    while True:
        try:
            query = input(bold("\n  Ask your question > ")).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {cyan('Goodbye! Have a great day.')}\n")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print(f"\n  {cyan('Goodbye! Have a great day.')}\n")
            break
        if query.lower() == "rebuild":
            rebuild_index()
            continue

        run_pipeline(query)


if __name__ == "__main__":
    main()

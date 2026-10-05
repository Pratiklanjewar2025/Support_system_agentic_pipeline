import argparse
import sys
import json
from evaluation.evaluator import run_evaluation

# Optional color printing if colorama is available
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    def cyan(s):    return Fore.CYAN    + str(s) + Style.RESET_ALL
    def green(s):   return Fore.GREEN   + str(s) + Style.RESET_ALL
    def yellow(s):  return Fore.YELLOW  + str(s) + Style.RESET_ALL
    def red(s):     return Fore.RED     + str(s) + Style.RESET_ALL
    def bold(s):    return Style.BRIGHT + str(s) + Style.RESET_ALL
except ImportError:
    def cyan(s):    return str(s)
    def green(s):   return str(s)
    def yellow(s):  return str(s)
    def red(s):     return str(s)
    def bold(s):    return str(s)

def main():
    parser = argparse.ArgumentParser(description="Terminal LLM Evaluator")
    parser.add_argument("--query", type=str, required=True, help="Technical support query to evaluate")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. gemini-3.5-flash)")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM Provider (gemini, groq, openai, etc)")
    
    args = parser.parse_args()
    
    print(cyan("\n" + "="*60))
    print(bold(f"  Single-Model Terminal Evaluation"))
    print(cyan("="*60))
    print(f"  Model    : {yellow(args.model)}")
    print(f"  Provider : {yellow(args.provider)}")
    print(f"  Query    : {args.query}")
    print(cyan("-" * 60))
    print(yellow("  Running Multi-Agent Pipeline... (This may take 30-60s)"))
    
    try:
        response = run_evaluation(query=args.query, model=args.model, provider=args.provider)
    except Exception as e:
        print(red(f"\n[ERROR] Pipeline crashed: {e}"))
        sys.exit(1)
        
    if response.status == "failed":
        print(red(f"\n[ERROR] {response.llm_metrics.accuracy.reason}"))
        sys.exit(1)
        
    print(green("  Evaluation Complete!\n"))
    
    # Print Metrics
    print(bold("  Overall Performance"))
    print(f"    Overall Score : {bold(cyan(response.overall_score))} / 100 ({response.quality_grade})")
    
    print(bold("\n  LLM Quality Metrics"))
    print(f"    Accuracy         : {green(response.llm_metrics.accuracy.score)}%  - {response.llm_metrics.accuracy.reason}")
    print(f"    Reasoning        : {green(response.llm_metrics.reasoning.score)}%  - {response.llm_metrics.reasoning.reason}")
    print(f"    Faithfulness     : {green(response.llm_metrics.faithfulness.score)}%  - {response.llm_metrics.faithfulness.reason}")
    print(f"    Hallucination    : {green(response.llm_metrics.hallucination.score)}%  - {response.llm_metrics.hallucination.reason}")
    print(f"    Response Content : {green(response.llm_metrics.response_content.score)}%  - {response.llm_metrics.response_content.reason}")
    
    print(bold("\n  Pipeline Metrics (Mathematical Formulas)"))
    print(f"    Context Overlap  : {cyan(response.pipeline_metrics.context_recall.score)}%  - {response.pipeline_metrics.context_recall.reason}")
    print(f"    Answer Richness  : {cyan(response.pipeline_metrics.answer_faithfulness.score)}%  - {response.pipeline_metrics.answer_faithfulness.reason}")
    print(f"    Token Throughput : {cyan('N/A')}  - {response.pipeline_metrics.task_success.reason}")
    print(f"    Tool Selection   : {cyan(response.pipeline_metrics.tool_selection_accuracy.score)}%  - {response.pipeline_metrics.tool_selection_accuracy.reason}")
    
    print(bold("\n  Resource Usage"))
    print(f"    Total Tokens     : {response.performance.total_tokens}")
    print(f"    End-to-End Time  : {response.performance.end_to_end_latency_ms / 1000:.2f}s")
    
    if response.final_answer:
        print(bold("\n  Model Generated Response"))
        print(cyan("-" * 60))
        print(f"  {response.final_answer}")
    
    print(cyan("\n" + "="*60 + "\n"))

if __name__ == "__main__":
    main()

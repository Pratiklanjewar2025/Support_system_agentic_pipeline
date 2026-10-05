import os
import json
import re
import argparse
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult, run_evaluator

# Load environment variables (API keys, LLM configs)
load_dotenv()

from agents.orchestrator import build_graph

@run_evaluator
def comprehensive_evaluator(run, example) -> dict:
    """
    Evaluates the correctness, reasoning, faithfulness, hallucination, and content quality 
    using the dynamic LLM configured in the environment, returning all metrics at once.
    """
    from agents.llm_utils import get_llm
    from langchain_core.messages import HumanMessage
    
    reference = example.outputs.get("answer") if example.outputs else "No reference"
    prediction = run.outputs.get("final_answer", "")
    kb_context = run.outputs.get("kb_context", "No context provided.")
    
    if not reference or reference == "No reference":
        return {"correctness": 0.0, "error": "No reference provided"}
        
    eval_llm = get_llm(temperature=0.0)
    prompt = (
        "You are an expert AI evaluator grading a technical support pipeline.\n"
        "Evaluate the response on the following 5 metrics on a scale of 0.0 to 1.0.\n\n"
        f"Context/Evidence retrieved: {kb_context}\n"
        f"Expected Answer: {reference}\n"
        f"Pipeline Output: {prediction}\n\n"
        "Metrics to evaluate:\n"
        "1. correctness: Is the final answer correct based on the expected answer? (1.0 = perfect, 0.0 = completely wrong)\n"
        "2. reasoning_quality: Is the reasoning logically valid and does it reach the right conclusion? (1.0 = excellent logic, 0.0 = flawed)\n"
        "3. faithfulness: Is the answer supported by the given Context/Evidence? (1.0 = fully supported, 0.0 = completely unsupported)\n"
        "4. hallucination_rate: How much unsupported/false information is generated? (0.0 = no hallucinations/best, 1.0 = entirely hallucinated)\n"
        "5. content_quality: Relevance, completeness, and clarity of the response. (1.0 = excellent, 0.0 = poor)\n\n"
        "Output ONLY valid JSON in the following format, with no markdown formatting or other text:\n"
        "{\n"
        '  "correctness": 0.9,\n'
        '  "reasoning_quality": 0.8,\n'
        '  "faithfulness": 1.0,\n'
        '  "hallucination_rate": 0.0,\n'
        '  "content_quality": 0.9\n'
        "}"
    )
    
    try:
        response = eval_llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        
        scores = json.loads(content)
        return {
            "correctness": float(scores.get("correctness", 0.0)),
            "reasoning_quality": float(scores.get("reasoning_quality", 0.0)),
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "hallucination_rate": float(scores.get("hallucination_rate", 1.0)),
            "content_quality": float(scores.get("content_quality", 0.0))
        }
    except Exception as e:
        print(f"Evaluation parsing error: {e}")
        return {
            "correctness": 0.0,
            "reasoning_quality": 0.0,
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            "content_quality": 0.0
        }

def pipeline_predictor(inputs: dict) -> dict:
    """
    The wrapper for the LangGraph pipeline used during evaluation.
    Takes inputs from LangSmith dataset and returns outputs for the evaluator.
    """
    graph = build_graph()
    
    query = inputs.get("query", "")
    if not query:
        query = inputs.get("question") or str(inputs)
        
    initial_state = {
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
    
    final_state = graph.invoke(initial_state)
    
    return {
        "final_answer": final_state.get("final_answer", ""),
        "kb_context": final_state.get("kb_context", ""),
        "category": final_state.get("category", ""),
        "escalated": final_state.get("escalated", False)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate the Multi-Agent Pipeline with LangSmith.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the LangSmith dataset to evaluate against.")
    parser.add_argument("--provider", type=str, help="LLM Provider (groq, openai, anthropic, gemini, openrouter, cerebras, huggingface)")
    parser.add_argument("--model", type=str, help="LLM Model Name")
    parser.add_argument("--api_key", type=str, help="API Key for the selected provider")
    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL_NAME"] = args.model
        
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if args.api_key:
        if provider == "groq":
            os.environ["GROQ_API_KEY"] = args.api_key
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = args.api_key
        elif provider == "gemini":
            os.environ["GEMINI_API_KEY"] = args.api_key
        elif provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = args.api_key
        elif provider == "cerebras":
            os.environ["CEREBRAS_API_KEY"] = args.api_key
        elif provider == "huggingface":
            os.environ["HUGGINGFACE_API_KEY"] = args.api_key

    client = Client()
    
    current_provider = os.getenv("LLM_PROVIDER", "groq")
    current_model = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")
    
    print(f"Starting comprehensive evaluation on dataset '{args.dataset}'")
    print(f"Using Provider: {current_provider.upper()} | Model: {current_model}")
    
    results = evaluate(
        pipeline_predictor,
        data=args.dataset,
        evaluators=[comprehensive_evaluator],
        experiment_prefix=f"TechSupport-{current_provider}-{current_model.replace('/', '-')}",
        metadata={
            "provider": current_provider,
            "model": current_model,
        }
    )
    
    print(f"\nEvaluation completed. Results can be viewed in the LangSmith UI.")

if __name__ == "__main__":
    main()

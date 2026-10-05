import os
import json
import re
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agents.orchestrator import build_graph
from agents.llm_utils import get_llm
from langchain_core.messages import HumanMessage

def evaluate_single_query(query: str):
    """
    Runs the pipeline on a single query and evaluates the results on 5 metrics
    without using a LangSmith dataset.
    """
    print(f"\n--- 1. Running Pipeline ---")
    graph = build_graph()
    
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
    
    prediction = final_state.get("final_answer", "")
    kb_context = final_state.get("kb_context", "No context provided.")
    
    print("\n[Pipeline Output]")
    print(prediction[:500] + "..." if len(prediction) > 500 else prediction)
    
    print(f"\n--- 2. Evaluating Response ---")
    eval_llm = get_llm(temperature=0.0)
    
    # Reference-free prompt (No expected answer needed!)
    prompt = (
        "You are an expert AI evaluator grading a technical support pipeline's response.\n"
        "Evaluate the response on the following 5 metrics on a scale of 0.0 to 1.0.\n\n"
        f"User Question: {query}\n"
        f"Context/Evidence retrieved by RAG: {kb_context}\n"
        f"Pipeline Output: {prediction}\n\n"
        "Metrics to evaluate:\n"
        "1. correctness: Does the Pipeline Output directly and correctly solve the User Question? (1.0 = perfect, 0.0 = completely wrong)\n"
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
    
    response = eval_llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    if isinstance(content, list):
        content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        
    content = content.strip()
    content = re.sub(r"^```[a-z]*\n?", "", content)
    content = re.sub(r"\n?```$", "", content)
    
    try:
        scores = json.loads(content)
        print("\n--- Final Metrics ---")
        for k, v in scores.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Failed to parse evaluation: {content}")
        print(e)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single query without a dataset.")
    parser.add_argument("--query", type=str, required=True, help="The technical support question to ask.")
    parser.add_argument("--provider", type=str, default="gemini", help="LLM Provider")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash-latest", help="LLM Model Name")
    args = parser.parse_args()

    # Set dynamic env variables
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["LLM_MODEL_NAME"] = args.model
    
    print(f"Provider: {args.provider.upper()} | Model: {args.model}")
    evaluate_single_query(args.query)

if __name__ == "__main__":
    main()

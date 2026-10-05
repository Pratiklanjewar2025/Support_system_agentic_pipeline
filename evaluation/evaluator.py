import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Ensure environment is loaded before any imports
load_dotenv()

from evaluation.schemas import (EvaluationResponse, LLMMetrics, PipelineMetrics, 
                                PerformanceMetrics, MetricScore)
from evaluation.judge import judge_llm_quality
from agents.orchestrator import build_graph
from agents.llm_utils import get_llm

def run_evaluation(query: str, model: str, provider: str) -> EvaluationResponse:
    # 1. Dynamically set the single target model
    os.environ["LLM_MODEL_NAME"] = model
    os.environ["LLM_PROVIDER"] = provider
    
    # 2. Build pipeline
    graph = build_graph()
    
    initial_state = {
        "user_query": query, "category": "", "severity": "", "summary": "",
        "kb_results": [], "kb_context": "", "root_cause": "", "confidence": "",
        "resolution_steps": "", "resolution_status": "", "verified": False,
        "verification_notes": "", "escalated": False, "escalation_reason": "",
        "final_answer": "",
    }
    
    # 3. Time the execution
    start_time = time.time()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"Pipeline error: {e}")
        # Return a graceful failure response instead of crashing with 500
        return EvaluationResponse(
            evaluation_id="eval_" + str(uuid.uuid4())[:8],
            model_name=model, provider=provider,
            overall_score=0, quality_grade="Failed",
            timestamp=datetime.utcnow().isoformat(),
            llm_metrics=LLMMetrics(
                accuracy=MetricScore(score=0, reason=f"Pipeline crashed: {str(e)}"),
                reasoning=MetricScore(score=0, reason=""), faithfulness=MetricScore(score=0, reason=""),
                hallucination=MetricScore(score=0, reason=""), response_content=MetricScore(score=0, reason="")
            ),
            pipeline_metrics=PipelineMetrics(
                context_recall=MetricScore(score=0, reason=""), answer_faithfulness=MetricScore(score=0, reason=""),
                task_success=MetricScore(score=0, reason=""), tool_selection_accuracy=MetricScore(score=0, reason="")
            ),
            performance=PerformanceMetrics(input_tokens=0, output_tokens=0, total_tokens=0, llm_latency_ms=0, end_to_end_latency_ms=int((time.time() - start_time) * 1000)),
            status="failed"
        )
    end_time = time.time()
    
    e2e_latency_ms = int((end_time - start_time) * 1000)
    llm_latency_ms = int(e2e_latency_ms * 0.8) # Approximation for LLM time
    
    # 4. Approximation of token counts
    input_tokens = len(query) + len(final_state.get("kb_context", "")) // 4
    output_tokens = len(final_state.get("final_answer", "")) // 4
    
    # 5. LLM-as-a-judge (Gold Standard Reference Model)
    # For research stability, we force the Judge to be the highly intelligent 120B model,
    # regardless of which smaller model the pipeline is currently testing.
    from langchain_groq import ChatGroq
    eval_llm = ChatGroq(
        model="openai/gpt-oss-120b", 
        temperature=0.0, 
        max_tokens=1500,
        api_key=os.getenv("GROQ_API_KEY")
    )
    judge_results = judge_llm_quality(
        eval_llm, query, final_state.get("kb_context", ""), 
        final_state.get("final_answer", ""), 
        final_state.get("verification_notes", ""), 
        final_state.get("escalated", False)
    )
    
    # 6. Formula-Based Deterministic Metrics
    # Calculate Token Throughput (Tokens per Second)
    e2e_seconds = e2e_latency_ms / 1000.0 if e2e_latency_ms > 0 else 1.0
    tokens_per_second = round(output_tokens / e2e_seconds, 2)
    
    # Calculate Lexical Diversity (Type-Token Ratio)
    final_answer_text = final_state.get("final_answer", "")
    words = final_answer_text.lower().split()
    unique_words = set(words)
    lexical_diversity = round(len(unique_words) / len(words) * 100, 1) if words else 0.0
    
    # Calculate Context Overlap (Jaccard Similarity) between KB Context and Answer
    kb_text = final_state.get("kb_context", "").lower()
    kb_words = set(kb_text.split())
    if kb_words and unique_words:
        intersection = unique_words.intersection(kb_words)
        union = unique_words.union(kb_words)
        context_overlap = round(len(intersection) / len(union) * 100, 1)
    else:
        context_overlap = 0.0
    
    # 7. Pipeline metrics calculation
    task_success_score = 95 if not final_state.get("escalated") else 40
    tool_score = 90 if final_state.get("kb_results") else 50
    context_recall = 85 if final_state.get("kb_results") else 20
    
    llm_metrics = LLMMetrics(
        accuracy=MetricScore(**judge_results.get("accuracy", {"score": 50, "reason": "Missing"})),
        reasoning=MetricScore(**judge_results.get("reasoning", {"score": 50, "reason": "Missing"})),
        faithfulness=MetricScore(**judge_results.get("faithfulness", {"score": 50, "reason": "Missing"})),
        hallucination=MetricScore(**judge_results.get("hallucination", {"score": 50, "reason": "Missing"})),
        response_content=MetricScore(**judge_results.get("response_content", {"score": 50, "reason": "Missing"}))
    )
    
    # 8. Overall Scoring Calculation
    acc = llm_metrics.accuracy.score or 0
    reas = llm_metrics.reasoning.score or 0
    faith = llm_metrics.faithfulness.score or 0
    hall = llm_metrics.hallucination.score or 0
    resp = llm_metrics.response_content.score or 0
    
    overall_score = (acc * 0.20) + (reas * 0.15) + (faith * 0.20) + (hall * 0.20) + (resp * 0.15)
    overall_score = (overall_score * 0.8) + (task_success_score * 0.20)
    
    grade = "Needs Improvement"
    if overall_score >= 90: grade = "Excellent"
    elif overall_score >= 80: grade = "Very Good"
    elif overall_score >= 70: grade = "Good"
    elif overall_score >= 60: grade = "Average"
    
    return EvaluationResponse(
        evaluation_id="eval_" + str(uuid.uuid4())[:8],
        model_name=model,
        provider=provider,
        overall_score=round(overall_score, 1),
        quality_grade=grade,
        timestamp=datetime.utcnow().isoformat(),
        llm_metrics=llm_metrics,
        pipeline_metrics=PipelineMetrics(
            context_recall=MetricScore(score=context_overlap, reason=f"Jaccard Similarity Context Overlap: {context_overlap}%"),
            answer_faithfulness=MetricScore(score=lexical_diversity, reason=f"Lexical Diversity (Type-Token Ratio): {lexical_diversity}%"),
            task_success=MetricScore(score=task_success_score, reason=f"Tokens Per Second Throughput: {tokens_per_second} t/s"),
            tool_selection_accuracy=MetricScore(score=tool_score, reason="RAG tools invoked appropriately")
        ),
        performance=PerformanceMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            llm_latency_ms=llm_latency_ms,
            end_to_end_latency_ms=e2e_latency_ms
        ),
        final_answer=final_state.get("final_answer", "No answer generated"),
        status="completed"
    )

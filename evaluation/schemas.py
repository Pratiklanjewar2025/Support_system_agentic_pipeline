from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class MetricScore(BaseModel):
    score: Optional[float]
    reason: Optional[str] = None
    status: str = "completed"

class LLMMetrics(BaseModel):
    accuracy: MetricScore
    reasoning: MetricScore
    faithfulness: MetricScore
    hallucination: MetricScore
    response_content: MetricScore

class PipelineMetrics(BaseModel):
    context_recall: MetricScore
    answer_faithfulness: MetricScore
    task_success: MetricScore
    tool_selection_accuracy: MetricScore

class PerformanceMetrics(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_latency_ms: int
    end_to_end_latency_ms: int

class EvaluationRequest(BaseModel):
    model: str
    query: Optional[str] = None
    dataset: Optional[str] = None
    provider: Optional[str] = "gemini"

class EvaluationResponse(BaseModel):
    evaluation_id: str
    model_name: str
    provider: str
    overall_score: float
    quality_grade: str
    timestamp: str
    llm_metrics: LLMMetrics
    pipeline_metrics: PipelineMetrics
    performance: PerformanceMetrics
    final_answer: Optional[str] = None
    status: str

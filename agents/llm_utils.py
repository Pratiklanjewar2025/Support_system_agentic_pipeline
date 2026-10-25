import os
import re
from typing import Optional, Callable, Any
from dotenv import load_dotenv
load_dotenv()
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Global mock hook for deterministic routing and workflow tests
_MOCK_LLM_ADAPTER = None


class MockDeterministicLLM(BaseChatModel):
    """
    Deterministic Mock LLM Adapter for Unit & Integration Testing of
    Multi-Agent Routing, Claim-Level Verification, and Budget Limits.
    Never replaces real inference in production unless explicitly activated via set_mock_llm.
    """
    response_map: Any = None
    default_response: str = '{"root_cause": "Diagnosed SOP configuration mismatch", "confidence": "high", "verified": true, "notes": "Claim verified"}'

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt_text = " ".join(m.content if hasattr(m, "content") else str(m) for m in messages)
        resp = self.default_response
        if isinstance(self.response_map, dict):
            for pattern, r in self.response_map.items():
                if re.search(pattern, prompt_text, re.I):
                    resp = r
                    break
        elif callable(self.response_map):
            resp = self.response_map(prompt_text)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resp))])

    @property
    def _llm_type(self) -> str:
        return "mock_deterministic_llm"


def set_mock_llm(mock_instance: Optional[BaseChatModel]):
    """Activates a deterministic mock LLM adapter for testing."""
    global _MOCK_LLM_ADAPTER
    _MOCK_LLM_ADAPTER = mock_instance


def clear_mock_llm():
    """Restores production LLM providers."""
    global _MOCK_LLM_ADAPTER
    _MOCK_LLM_ADAPTER = None


def get_mock_llm() -> Optional[BaseChatModel]:
    global _MOCK_LLM_ADAPTER
    return _MOCK_LLM_ADAPTER


def estimate_token_cost(input_tokens: int, output_tokens: int, model: str = None, provider: str = None) -> float:
    """Computes exact API dollar cost based on model tier pricing per 1K tokens."""
    model_str = (model or "").lower()
    # All OpenRouter free-tier models cost $0.00
    if ":free" in model_str or "gemma-4" in model_str and ":free" in model_str:
        cost_per_1k = 0.00000
    elif "nemotron" in model_str and ":free" in model_str:
        cost_per_1k = 0.00000
    elif "inkling" in model_str and ":free" in model_str:
        cost_per_1k = 0.00000
    elif "north-mini" in model_str and ":free" in model_str:
        cost_per_1k = 0.00000
    elif "claude" in model_str or "sonnet" in model_str or "haiku" in model_str:
        cost_per_1k = 0.00100
    elif "deepseek" in model_str:
        cost_per_1k = 0.00020
    elif "120b" in model_str or "command-r-plus" in model_str or "pro" in model_str:
        cost_per_1k = 0.00250
    elif "flash" in model_str or "70b" in model_str:
        cost_per_1k = 0.00015
    elif "command-r" in model_str:
        cost_per_1k = 0.00015
    else:  # Fast tier (Groq 20B, 8B, etc.)
        cost_per_1k = 0.00010

    total_tokens = input_tokens + output_tokens
    return round((total_tokens / 1000.0) * cost_per_1k, 7)


def get_llm(temperature=0.0, max_tokens=300, model: str = None, provider: str = None) -> BaseChatModel:
    """
    Returns an LLM instance dynamically based on arguments or environment variables.
    Supported providers: groq, openai, anthropic, gemini, openrouter, cerebras, huggingface, deepseek, cohere
    """
    global _MOCK_LLM_ADAPTER
    if _MOCK_LLM_ADAPTER is not None:
        return _MOCK_LLM_ADAPTER

    # Detect default provider based on available keys if not provided
    if not provider:
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider:
            provider = env_provider.lower()
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("COHERE_API_KEY"):
            provider = "cohere"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            provider = "groq"
    else:
        provider = provider.lower()


    # Determine appropriate default model per provider
    default_models = {
        "groq": "openai/gpt-oss-20b",
        "gemini": "gemini-2.5-flash",
        "google": "gemini-2.5-flash",
        "cohere": "command-r-08-2024",
        "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
    }
    
    env_model = os.getenv("LLM_MODEL_NAME")
    if model:
        model_name = model
    elif env_model and env_model not in ("llama3-8b-8192", "google/gemma-4-31b-it:free"):
        model_name = env_model
    else:
        model_name = default_models.get(provider, "openai/gpt-oss-20b")
    
    # Catch decommissioned groq model
    if provider == "groq" and model_name in ("llama3-8b-8192", "llama3-8b"):
        model_name = "openai/gpt-oss-20b"
        
    if provider in ("gemini", "google") and model_name in ("gemini-1.5-flash", "llama3-8b-8192"):
        model_name = "gemini-2.5-flash"
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        return ChatGroq(model=model_name, temperature=temperature, max_tokens=max(max_tokens, 500), api_key=api_key)
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max(max_tokens, 500), api_key=api_key)
        
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max(max_tokens, 500), api_key=api_key)
        
    elif provider in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max(max_tokens, 1500), google_api_key=api_key)
        
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max(max_tokens, 500),
            default_headers={"HTTP-Referer": "https://github.com/adarag-ops", "X-Title": "AdaRAG-Ops"}
        )
        
    elif provider == "cerebras":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable is missing.")
        return ChatOpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY or HF_TOKEN environment variable is missing.")
            
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            task="text-generation",
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.01,
            huggingfacehub_api_token=api_key,
        )
        return ChatHuggingFace(llm=llm)
        
    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is missing.")
        return ChatOpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key,
            model=model_name if model_name and "llama" not in model_name else "deepseek-chat",
            temperature=temperature,
            max_tokens=max(max_tokens, 500),
        )
        
    elif provider == "cohere":
        from langchain_cohere import ChatCohere
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is missing.")
        return ChatCohere(
            cohere_api_key=api_key,
            model=model_name if model_name and "llama" not in model_name else "command-r-08-2024",
            temperature=temperature,
            max_tokens=max(max_tokens, 500),
            request_timeout=35.0,
        )
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

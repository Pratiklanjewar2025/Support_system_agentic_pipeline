import os
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(temperature=0.0, max_tokens=300) -> BaseChatModel:
    """
    Returns an LLM instance dynamically based on environment variables.
    Supported providers: groq, openai, anthropic, gemini, openrouter, cerebras, huggingface
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model_name = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing.")
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key)
        
    elif provider in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is missing.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens, google_api_key=api_key)
        
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
            max_tokens=max_tokens,
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
        
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

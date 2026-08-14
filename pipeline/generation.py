"""
Generation Module

Handles LLM answer generation using Groq.
"""

import time
import logging
from typing import List, Tuple
import httpx
import json

from .schemas import Evidence, GenerationRequest, GenerationResponse
from config import settings

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Custom exception for generation errors."""
    pass


def build_context(query: str, evidence: List[Evidence]) -> str:
    """
    Build context string for LLM from evidence.
    
    Args:
        query: User query
        evidence: List of Evidence objects
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, e in enumerate(evidence, 1):
        context_parts.append(f"Evidence {i}: {e.passage}")
    
    context = "\n\n".join(context_parts)
    
    return f"""
    Query: {query}

    Context:
    {context}

    Instructions:
    - Answer the query using ONLY the provided context
    - If you cannot answer from the context, say "I don't know"
    - Be concise and factual
    - Cite specific evidence when possible
    """


async def generate_answer(
    query: str,
    context: str,
    language: str = "en",
) -> str:
    """
    Generate answer using Groq LLM.
    
    Args:
        query: User query
        context: Context string
        language: Language for response
        
    Returns:
        Generated answer text
        
    Raises:
        GenerationError: If answer generation fails
    """
    start_time = time.perf_counter()
    
    try:
        # Get Groq API settings
        api_key = settings.GROQ_API_KEY
        model = settings.GROQ_LLM_MODEL
        
        if not api_key:
            raise GenerationError("GROQ_API_KEY not configured")
        if not model:
            raise GenerationError("GROQ_LLM_MODEL not configured")
        
        # Prepare prompt with structured output requirement
        system_prompt = """You are a factual question-answering assistant. 
        Answer questions based ONLY on the provided context. 
        If you cannot answer from the context, respond with: "I don't have enough information to answer this question."
        Be concise, factual, and cite evidence when possible."""
        
        user_prompt = f"""
        Query: {query}

        Context:
        {context}

        Answer:
        """
        
        # Prepare request
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "response_format": {"type": "text"},
        }
        
        # Send request to Groq
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Groq LLM error: {response.status_code} - {response.text}")
                raise GenerationError(f"Groq LLM failed: {response.status_code} - {response.text}")
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"LLM generation completed in {latency_ms:.2f}ms")
            
            return answer
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during generation: {e}", exc_info=True)
        raise GenerationError(f"HTTP error: {e}")
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise GenerationError(f"Generation failed: {e}")

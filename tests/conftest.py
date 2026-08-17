"""
Pytest Configuration

Shared fixtures and configuration for tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import httpx

from config import settings
from retrieval import QdrantStore, BM25Store


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("config.settings") as mock_settings:
        mock_settings.GROQ_API_KEY = "test_api_key"
        mock_settings.GROQ_STT_MODEL = "whisper-large-v3-turbo"
        mock_settings.GROQ_LLM_MODEL = "llama-3.1-70b-versatile"
        mock_settings.QDRANT_URL = "http://localhost:6333"
        mock_settings.QDRANT_API_KEY = "test_qdrant_key"
        mock_settings.QDRANT_COLLECTION = "test_collection"
        mock_settings.APP_ENV = "test"
        yield mock_settings


@pytest.fixture
async def mock_qdrant_store():
    """Mock QdrantStore for testing."""
    store = QdrantStore()
    store.client = AsyncMock()
    
    # Mock search response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": [
            {
                "id": 1,
                "score": 0.9,
                "payload": {
                    "passage": "Test passage",
                    "language": "en",
                    "metadata": {},
                },
            },
        ]
    }
    
    store.client.post = AsyncMock(return_value=mock_response)
    store.client.put = AsyncMock(return_value=mock_response)
    
    yield store


@pytest.fixture
async def mock_bm25_store():
    """Mock BM25Store for testing."""
    store = BM25Store()
    
    # Add test documents
    test_docs = [
        {"id": "1", "passage": "Test passage one", "language": "en"},
        {"id": "2", "passage": "Test passage two", "language": "en"},
    ]
    
    await store.add_documents(test_docs)
    
    yield store


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

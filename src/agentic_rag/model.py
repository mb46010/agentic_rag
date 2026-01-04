# src/agentic_rag/model.py

import logging

from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


def get_default_model():
    model = init_chat_model(
        model="gpt-4.1",
        temperature=0.0,
        max_tokens=5000,
    )
    return model

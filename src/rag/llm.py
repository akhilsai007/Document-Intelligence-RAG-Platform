"""Pluggable LLM client. Providers: stub | groq | openai | bedrock.

The stub provider needs no credentials: it composes a grounded, extractive
answer from the retrieved context so the full pipeline is demonstrable
offline. Swap LLM_PROVIDER to use a real model in production. Groq offers very
low-latency inference for open models (Llama, Mixtral, Gemma) via an
OpenAI-compatible chat API.
"""
from __future__ import annotations

from typing import List

from config.settings import settings

SYSTEM_PROMPT = (
    "You are a document intelligence assistant. Answer the question using only "
    "the provided context. Cite sources as [doc_id]. If the context is "
    "insufficient, say so."
)


def _build_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = provider or settings.llm_provider

    def generate(self, question: str, contexts: List[str], sources: List[str]) -> str:
        if self.provider == "groq":
            return self._groq(question, contexts)
        if self.provider == "openai":
            return self._openai(question, contexts)
        if self.provider == "bedrock":
            return self._bedrock(question, contexts)
        return self._stub(question, contexts, sources)

    # --- providers ---
    def _groq(self, question: str, contexts: List[str]) -> str:  # pragma: no cover
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(question, contexts)},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    def _stub(self, question: str, contexts: List[str], sources: List[str]) -> str:
        if not contexts:
            return "I couldn't find anything relevant in the indexed documents."
        top = contexts[0].strip()
        snippet = top[:500] + ("…" if len(top) > 500 else "")
        cited = ", ".join(f"[{s}]" for s in dict.fromkeys(sources))
        return (
            f"Based on the retrieved documents, here is the most relevant "
            f"information:\n\n{snippet}\n\nSources: {cited}"
        )

    def _openai(self, question: str, contexts: List[str]) -> str:  # pragma: no cover
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(question, contexts)},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content

    def _bedrock(self, question: str, contexts: List[str]) -> str:  # pragma: no cover
        import json

        import boto3

        client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": _build_prompt(question, contexts)}
            ],
        }
        resp = client.invoke_model(modelId=settings.bedrock_model_id, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]

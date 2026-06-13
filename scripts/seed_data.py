"""Generate router training data for the document corpus.

This version is configured for a research-paper corpus across four ML topics:
RAG, NLP, Reinforcement Learning, and LLMs. It writes only the router training
phrases (and an empty labels file so ingestion auto-classifies each paper via
the trained router). It does NOT create any sample documents — your own PDFs in
data/sample_docs are the corpus.
"""
from __future__ import annotations

import json
import os

# Short, query-like labeled phrases for training the router. The router
# classifies the incoming *question* at query time, so training on phrases that
# look like questions (and use category-distinct vocabulary) works best.
ROUTER_PHRASES = {
    "rag": [
        "retrieval augmented generation", "vector store retrieval",
        "dense passage retrieval", "retrieve relevant chunks",
        "grounding answers in documents", "retriever and generator pipeline",
        "knowledge base lookup for answers", "top k document retrieval",
        "embedding based retrieval", "retrieval index for question answering",
        "cite retrieved source passages", "context retrieval for generation",
        "augment the prompt with retrieved text", "retrieval pipeline for qa",
        "chunking and indexing documents",
    ],
    "nlp": [
        "named entity recognition", "part of speech tagging",
        "sentiment analysis of text", "machine translation model",
        "text summarization task", "syntactic dependency parsing",
        "coreference resolution", "sequence labeling task",
        "text classification benchmark", "tokenization and lemmatization",
        "semantic textual similarity", "language understanding benchmark",
        "word sense disambiguation", "natural language inference",
        "information extraction from text",
    ],
    "reinforcement-learning": [
        "reward function design", "policy gradient method",
        "q learning algorithm", "markov decision process",
        "exploration exploitation tradeoff", "actor critic architecture",
        "value function estimation", "reinforcement learning agent",
        "temporal difference learning", "proximal policy optimization",
        "deep q network", "bellman equation update",
        "on policy versus off policy", "reward shaping for an agent",
        "environment state action reward",
    ],
    "llm": [
        "pretraining large language models", "neural scaling laws",
        "instruction tuning a model", "in context learning",
        "emergent abilities of large models", "next token prediction objective",
        "supervised fine tuning", "reinforcement learning from human feedback",
        "decoder only transformer", "model alignment and safety",
        "parameter efficient fine tuning", "prompt engineering techniques",
        "billions of model parameters", "pretraining corpus and tokens",
        "chain of thought prompting",
    ],
}


def main(
    out_dir: str = "data/sample_docs",
    labels_path: str = "data/labels.json",
    router_path: str = "data/router_training.json",
):
    os.makedirs(out_dir, exist_ok=True)

    # No sample documents are created — the corpus is your own PDFs in out_dir.
    # Write an empty labels map so ingestion auto-classifies each document with
    # the trained router instead of using hardcoded labels.
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({}, f)

    router_examples = [
        {"text": phrase, "category": cat}
        for cat, phrases in ROUTER_PHRASES.items()
        for phrase in phrases
    ]
    with open(router_path, "w", encoding="utf-8") as f:
        json.dump(router_examples, f, indent=2)

    print(f"Wrote {len(router_examples)} router training phrases across "
          f"{len(ROUTER_PHRASES)} categories to {router_path} "
          f"(no sample docs created; corpus = your PDFs in {out_dir})")


if __name__ == "__main__":
    main()
from src.rag import get_pipeline
from src.vectorstore import get_vector_store


def pipeline_dep():
    return get_pipeline()


def store_dep():
    return get_vector_store()

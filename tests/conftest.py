import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_artifacts():
    """Point all artifact paths at a temp dir so tests never touch real state."""
    tmp = tempfile.mkdtemp(prefix="rag-test-")
    os.environ["VECTOR_STORE_PATH"] = os.path.join(tmp, "vs")
    os.environ["ROUTER_MODEL_PATH"] = os.path.join(tmp, "router.joblib")
    os.environ["METADATA_LOCAL_PATH"] = os.path.join(tmp, "meta.sqlite")
    os.environ["MLFLOW_TRACKING_URI"] = os.path.join(tmp, "mlruns")
    os.environ["EMBEDDING_BACKEND"] = "hashing"
    os.environ["LLM_PROVIDER"] = "stub"
    yield

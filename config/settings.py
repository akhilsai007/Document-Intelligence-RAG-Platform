"""Central configuration. All values have local-friendly defaults so the
platform runs end-to-end with zero external services configured."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "stub"            # stub | groq | openai | bedrock
    openai_api_key: str = ""
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embeddings
    embedding_backend: str = "hashing"    # hashing | sentence-transformers
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Vector store
    vector_store_path: str = "./artifacts/vectorstore"

    # Router
    router_model_path: str = "./artifacts/router/router.joblib"

    # Snowflake / metadata
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_password: str = ""
    snowflake_warehouse: str = ""
    snowflake_database: str = ""
    snowflake_schema: str = ""
    metadata_local_path: str = "./artifacts/metadata.sqlite"

    # AWS / MLflow
    aws_region: str = "us-east-1"
    mlflow_tracking_uri: str = "./artifacts/mlruns"

    # Retrieval
    top_k: int = 4

    @property
    def snowflake_enabled(self) -> bool:
        return bool(self.snowflake_account and self.snowflake_user)


settings = Settings()

"""PySpark ingestion job: parse -> chunk -> embed in parallel across a
cluster, then write chunk records to Parquet. This is the production path
for embedding millions of documents. For local/dev use scripts/ingest.py.

Run (cluster):
    spark-submit src/ingestion/spark_embed.py \
        --input s3://bucket/raw-docs --output s3://bucket/chunks.parquet
"""
from __future__ import annotations

import argparse


def build_spark(app_name: str = "rag-ingest"):
    from pyspark.sql import SparkSession

    return SparkSession.builder.appName(app_name).getOrCreate()


def run(input_path: str, output_path: str) -> None:
    from pyspark.sql import Row

    from src.embeddings import Embedder
    from src.ingestion.chunker import chunk_text

    spark = build_spark()
    sc = spark.sparkContext

    # wholeTextFiles -> (path, content); works for text/markdown corpora
    raw = sc.wholeTextFiles(input_path)

    def to_chunks(record):
        path, content = record
        doc_id = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for i, ch in enumerate(chunk_text(content)):
            yield (f"{doc_id}::{i}", doc_id, ch)

    chunk_rdd = raw.flatMap(to_chunks)

    def embed_partition(rows):
        # one embedder per partition (amortizes model load across the partition)
        embedder = Embedder()
        rows = list(rows)
        if not rows:
            return iter([])
        vectors = embedder.encode([r[2] for r in rows])
        out = []
        for (cid, did, text), vec in zip(rows, vectors):
            out.append(Row(chunk_id=cid, doc_id=did, text=text, embedding=vec.tolist()))
        return iter(out)

    embedded = chunk_rdd.mapPartitions(embed_partition)
    df = spark.createDataFrame(embedded)
    df.write.mode("overwrite").parquet(output_path)
    spark.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input path (local dir, HDFS, or s3://)")
    p.add_argument("--output", required=True, help="Output Parquet path")
    args = p.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()

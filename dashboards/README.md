# BI dashboards (Power BI / Tableau)

The dashboards read from the metadata layer (`doc_metadata`, `query_log`). In
production these are Snowflake tables; locally they are SQLite tables with the
same schema (`src/metadata/snowflake_client.py`).

## Suggested Snowflake views

```sql
-- Corpus composition by category
CREATE OR REPLACE VIEW v_corpus_by_category AS
SELECT category, COUNT(*) AS chunk_count, COUNT(DISTINCT doc_id) AS doc_count
FROM doc_metadata
GROUP BY category;

-- Query volume and latency over time
CREATE OR REPLACE VIEW v_query_trends AS
SELECT DATE_TRUNC('hour', ts) AS hour,
       category,
       COUNT(*)            AS queries,
       AVG(latency_ms)     AS avg_latency_ms,
       AVG(n_results)      AS avg_results
FROM query_log
GROUP BY 1, 2;

-- Retrieval quality proxy: share of queries that returned >= 1 result
CREATE OR REPLACE VIEW v_answer_coverage AS
SELECT category,
       AVG(CASE WHEN n_results > 0 THEN 1 ELSE 0 END) AS coverage_rate
FROM query_log
GROUP BY category;
```

## Connecting

- **Power BI**: Get Data -> Snowflake -> enter account/warehouse -> select the
  `v_*` views -> build visuals. Suggested pages: Corpus Overview (category mix),
  Usage (queries/latency over time), Quality (coverage rate, avg results).
- **Tableau**: Connect -> Snowflake -> point at the same views. Use the
  `v_query_trends.hour` field on the date shelf for time-series.

For the local SQLite fallback, use a SQLite ODBC driver or export the tables to
CSV (`sqlite3 artifacts/metadata.sqlite .dump`) for a quick demo.

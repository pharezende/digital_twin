from prometheus_client import Counter, Gauge, Histogram


RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
    ["status"],
)

RAG_PROCESSING_DURATION = Histogram(
    "rag_processing_duration_seconds",
    "RAG processing duration from question input to generated response",
    buckets=(
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        30.0,
    ),
)

RAG_RESPONSE_CHARACTERS = Histogram(
    "rag_response_characters",
    "Number of characters in generated response",
    buckets=(
        50,
        100,
        250,
        500,
        1000,
        2000,
        5000,
    ),
)

VECTOR_STORE_DOCUMENTS = Gauge(
    "vector_store_documents",
    "Number of chunks currently stored in Chroma",
)

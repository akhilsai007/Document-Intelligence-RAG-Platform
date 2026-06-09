from src.ingestion.chunker import chunk_text


def test_empty():
    assert chunk_text("") == []


def test_overlap_and_coverage():
    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(words, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # consecutive chunks share the overlap region
    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)

import pytest

from agentic_rag_assistant.chunking import chunk_text


@pytest.mark.parametrize(
    ("length", "expected_spans"),
    [
        (0, []),
        (1, [(0, 1)]),
        (1_000, [(0, 1_000)]),
        (1_001, [(0, 1_000), (800, 1_001)]),
        (1_800, [(0, 1_000), (800, 1_800)]),
        (1_801, [(0, 1_000), (800, 1_800), (1_600, 1_801)]),
    ],
)
def test_chunk_boundaries_without_redundant_tail(length, expected_spans) -> None:
    text = "x" * length
    chunks = chunk_text(text)

    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == expected_spans
    assert all(chunk.text == text[chunk.start_char : chunk.end_char] for chunk in chunks)


def test_overlap_preserves_unicode_and_exact_source_text() -> None:
    text = "αβγδεζηθικ"
    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["αβγδεζ", "εζηθικ"]
    assert [(chunk.start_char, chunk.end_char) for chunk in chunks] == [(0, 6), (4, 10)]
    assert chunks[0].text + chunks[1].text[2:] == text


def test_whitespace_is_preserved_but_empty_windows_are_skipped() -> None:
    assert chunk_text(" \n\t ") == []
    assert chunk_text(" \nhello\t ")[0].text == " \nhello\t "


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (10, -1), (10, 10)])
def test_invalid_chunk_configuration_is_rejected(size, overlap) -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=size, overlap=overlap)

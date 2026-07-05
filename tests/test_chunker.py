from src.tts.chunker import HARD_LIMIT, achunk_stream, chunk_stream


def test_sentences_flush_as_they_complete():
    tokens = ["Hey. ", "I'm Rababb", ". I study ", "at Waterloo. Ask me stuff"]
    assert list(chunk_stream(tokens)) == [
        "Hey.",
        "I'm Rababb.",
        "I study at Waterloo.",
        "Ask me stuff",
    ]


def test_question_and_exclamation_boundaries():
    assert list(chunk_stream(["Really? ", "Yes! ", "Cool"])) == ["Really?", "Yes!", "Cool"]


def test_long_runon_breaks_at_clause():
    text = "I worked on perception for the car, " * 4  # no sentence end at all
    chunks = list(chunk_stream([text]))
    assert len(chunks) > 1
    assert all(len(c) <= HARD_LIMIT + 40 for c in chunks)


def test_no_boundary_hard_flush():
    text = "word " * 100
    chunks = list(chunk_stream([text]))
    assert len(chunks) > 1
    assert " ".join(chunks).split() == text.split()


def test_empty_stream():
    assert list(chunk_stream([])) == []
    assert list(chunk_stream(["   "])) == []


async def test_async_matches_sync():
    tokens = ["One. ", "Two, and ", "three. ", "Done"]

    async def agen():
        for t in tokens:
            yield t

    got = [c async for c in achunk_stream(agen())]
    assert got == list(chunk_stream(tokens))

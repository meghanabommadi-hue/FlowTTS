"""Unit tests for the streaming text splitter (stdlib only — no GPU/torch/numpy).

Run:  python -m flowtts.test.test_text_chunker
  or: python -m pytest flowtts/test/test_text_chunker.py -q
"""

from flowtts.synthesis.text_chunker import normalize_text, split_for_streaming


def test_empty():
    assert split_for_streaming("") == []
    assert split_for_streaming("   ") == []


def test_single_short():
    out = split_for_streaming("Hello there.", first_chunk_max_chars=60, chunk_max_chars=160)
    assert out == ["Hello there."]


def test_first_chunk_is_short():
    text = ("First short sentence here. Then a considerably longer follow up sentence "
            "that carries the bulk of the content and should land in a later chunk.")
    out = split_for_streaming(text, first_chunk_max_chars=30, chunk_max_chars=200)
    assert len(out) >= 2
    assert len(out[0]) <= 30 + 1  # first chunk capped (single sentence may slightly exceed)
    assert " ".join(out).split() == text.split()  # no words lost / reordered


def test_hindi_danda_split():
    text = "नमस्ते। मैं प्रिया बोल रही हूँ। आपकी EMI बारह सौ रुपए है।"
    out = split_for_streaming(text, first_chunk_max_chars=20, chunk_max_chars=40)
    assert len(out) >= 2
    # every danda-terminated sentence's text is preserved across the chunks
    assert "प्रिया" in " ".join(out)


def test_chunks_respect_cap():
    text = ". ".join(f"sentence number {i} with some filler words" for i in range(20))
    out = split_for_streaming(text, first_chunk_max_chars=40, chunk_max_chars=80)
    for c in out[1:]:
        # non-first chunks should not wildly exceed the cap (allow one word overflow)
        assert len(c) <= 80 + 40, f"chunk too long ({len(c)}): {c!r}"
    assert len(out) >= 3


def test_tiny_trailing_merged():
    out = split_for_streaming("A big first chunk of words here. ok",
                              first_chunk_max_chars=40, chunk_max_chars=40, min_chunk_chars=12)
    assert out[-1] != "ok"  # merged into predecessor
    assert "ok" in out[-1]


def test_normalize_text():
    assert normalize_text("Hello 😀 नमस्ते ☺️") == "Hello  नमस्ते"
    assert normalize_text("مرحبا hi") == "hi"          # Arabic stripped, ASCII kept
    assert normalize_text("  spaced  ") == "spaced"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run_all()

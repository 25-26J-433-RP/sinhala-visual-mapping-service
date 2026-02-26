from sinhala_normalization import normalize_sinhala_text


def test_spelling_correction_common_forms():
    text = "මෙය සමග සහ පිලිබඳව විස්තරයකි"
    out = normalize_sinhala_text(text)
    assert "සමඟ" in out
    assert "පිළිබඳව" in out


def test_token_boundary_recovery_glued_function_words():
    text = "අධ්‍යාපනයසඳහාවැදගත් දැනුම"
    out = normalize_sinhala_text(text)
    # boundary before the function word should be recovered
    assert "සඳහා" in out
    tokens = out.split()
    assert any(tok.startswith("අධ්") for tok in tokens)


def test_token_boundary_recovery_sinhala_latin_transition():
    text = "AIසහමානව"
    out = normalize_sinhala_text(text)
    # should insert boundaries around Sinhala/Latin transitions
    assert "AI" in out
    assert "සහ" in out

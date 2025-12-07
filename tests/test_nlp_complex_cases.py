import pytest

from mindmap_generator import SinhalaMindMapGenerator


@pytest.fixture()
def gen():
    return SinhalaMindMapGenerator()


def test_zero_width_joiner_handled(gen):
    # Contains zero-width joiner and non-visible characters
    text = "ශ්‍රී\u200Dලංකාව\u200D සුන්දරයි. එය වෙරළයි."
    res = gen.generate(text)
    assert isinstance(res, dict)
    assert res['nodes'] and res['metadata']['total_nodes'] > 0


def test_punctuation_variants_sentence_splitting(gen):
    text = "මෙය පරීක්ෂණයකි: එය වැඩ කරයි! කෙටිය? නියමයි। අන්තිමයි"
    sentences = gen._split_into_sentences(text)
    # Expect several sentences and no very short fragments
    assert len(sentences) >= 3
    assert all(len(s) > 5 for s in sentences)


def test_abbreviation_does_not_produce_short_fragments(gen):
    # 'ශ්‍රී.' is a common abbreviation; ensure short fragments are filtered out
    text = "ශ්‍රී. ලංකාව දිවයිනකි. එය සුන්දරයි."
    sentences = gen._split_into_sentences(text)
    # The short 'ශ්‍රී' fragment should be filtered (length <=5)
    assert all(len(s) > 5 for s in sentences)


def test_long_text_generation_stable(gen):
    # Very long input (repeated sentences) should not create excessive nodes
    sentence = "ශ්‍රී ලංකාව සුන්දරයි. "
    long_text = sentence * 5000
    res = gen.generate(long_text)
    # Implementation creates limited hierarchical nodes per paragraph, so expect node count small
    assert res['metadata']['total_nodes'] < 1000


def test_nonstandard_whitespace_and_zero_width_spaces(gen):
    text = "ශ්‍රී\u200B ලංකාව\tදකුණු\nආසියාවේ පවතී. එය සුන්දරයි."
    res = gen.generate(text)
    assert res['metadata']['total_nodes'] > 0


def test_mixed_language_and_urls_and_emojis(gen):
    text = "ශ්‍රී ලංකාව is beautiful. Visit http://example.com 😊. එය සුන්දරයි."
    res = gen.generate(text)
    # Ensure generator handles mixed-language input and truncates labels appropriately
    for n in res['nodes']:
        assert 'label' in n
        assert len(n['label']) <= 80  # root truncation uses 80 by default


def test_complex_keyphrase_splitting(gen):
    sentence = "මෙය (සිතුවිලි), සහ අංශ, උදාහරණය; තවත් කොටස"
    phrases = gen._extract_key_phrases(sentence)
    # Should extract at least one meaningful phrase, respecting length bounds
    assert len(phrases) >= 1
    for p in phrases:
        assert 10 <= len(p) <= 50

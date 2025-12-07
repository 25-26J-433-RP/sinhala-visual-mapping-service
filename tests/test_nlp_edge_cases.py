import pytest

from mindmap_generator import SinhalaMindMapGenerator


@pytest.fixture()
def gen():
    return SinhalaMindMapGenerator()


def test_split_into_sentences_handles_various_delimiters(gen):
    text = "මෙය පරීක්ෂණයකි: එය වැඩ කරයි! කෙටියේ? නියමයි। මුළුකථාව\nනවපේළියක්"
    sentences = gen._split_into_sentences(text)
    # Should split into meaningful sentences (filtering out very short fragments)
    assert any('පරීක්ෂණයකි' in s for s in sentences)
    assert any('වැඩ කරයි' in s for s in sentences)


def test_split_into_sentences_filters_short_fragments(gen):
    text = "A. කෙටිය. B. C."  # some very short fragments mixed in
    # Use Sinhala fragments long enough and short ones
    sentences = gen._split_into_sentences(text)
    # The method only keeps sentences longer than 5 chars
    assert all(len(s) > 5 for s in sentences)


def test_split_into_paragraphs_splits_on_double_newline(gen):
    text = "පරාග්‍රාෆ් එක.\n\nපරාග්‍රාෆ් දෙක."
    paras = gen._split_into_paragraphs(text)
    assert len(paras) == 2
    assert 'පරාග්‍රාෆ් එක' in paras[0]


def test_extract_key_phrases_with_commas_and_length(gen):
    sentence = "මෙම වගුව, විශාල පෙළක් ඇති පරාග්‍රාෆ් එක, අනෙක් කොටස"  # several comma parts
    phrases = gen._extract_key_phrases(sentence)
    # Should return parts with length between 10 and 50 characters
    assert any(',' not in p for p in phrases)
    for p in phrases:
        assert 10 <= len(p) <= 50


def test_extract_key_phrases_fallback_for_long_sentence(gen):
    sentence = "මෙය දිගු වාක්‍යයක් වන අතර කිසිදු කොමා නොමැති බැවින් මුලික වචන ගණනක් ගනී"
    phrases = gen._extract_key_phrases(sentence)
    # Fallback should return at least one phrase built from first few words
    assert len(phrases) >= 1
    assert phrases[0].split()[0] in sentence


def test_truncate_text_short_and_long(gen):
    short = "කෙටි ටෙක්ස්ට්"
    assert gen._truncate_text(short, 50) == short

    long_text = "x" * 100
    truncated = gen._truncate_text(long_text, 20)
    assert len(truncated) <= 20
    assert truncated.endswith('...')


def test_extract_main_topic_prefers_topic_keyword(gen):
    text = "විෂය ශ්‍රී ලංකාව පිළිබඳ විස්තර"  # contains 'විෂය' keyword
    main = gen._extract_main_topic(text)
    assert 'ශ්‍රී ලංකාව' in main

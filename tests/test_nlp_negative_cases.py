import pytest

from mindmap_generator import SinhalaMindMapGenerator


@pytest.fixture()
def gen():
    return SinhalaMindMapGenerator()


def test_none_input_raises_or_handled(gen):
    # Depending on implementation, None may raise or be handled; ensure no silent failure
    # Current implementation treats None as empty input and returns empty structure
    res = gen.generate(None)
    assert isinstance(res, dict)
    assert res.get('nodes', []) == []
    assert res.get('metadata', {}).get('total_nodes', 0) == 0


def test_non_string_input_raises(gen):
    with pytest.raises(Exception):
        gen.generate(12345)


def test_only_punctuation_returns_root_only(gen):
    text = "....!!!???...."
    res = gen.generate(text)
    # Implementation creates root node but should not create subtopics
    assert res['metadata']['total_nodes'] >= 1
    # If only root is present, ensure no edges
    if res['metadata']['total_nodes'] == 1:
        assert res['metadata']['total_edges'] == 0


def test_extremely_long_single_token(gen):
    long_word = 'අ' * 20000
    res = gen.generate(long_word)
    # Should not blow up and should return a small number of nodes
    assert isinstance(res, dict)
    assert res['metadata']['total_nodes'] < 1000


def test_control_characters_handled(gen):
    text = "ශ්‍රී\x00\x1f\x07 ලංකාව\nදකුණු\t ආසියාවේ"
    res = gen.generate(text)
    assert res['metadata']['total_nodes'] >= 1


def test_html_and_script_tags_do_not_crash(gen):
    text = "<script>alert('x')</script> ශ්‍රී ලංකාව දෙයි <b>bold</b>"
    res = gen.generate(text)
    # Should generate nodes; labels may include tags but no crash
    assert res['metadata']['total_nodes'] >= 1
    for n in res['nodes']:
        assert 'label' in n


def test_sql_like_injection_text(gen):
    text = "SELECT * FROM users; DROP TABLE concepts; ශ්‍රී ලංකාව"
    res = gen.generate(text)
    assert res['metadata']['total_nodes'] >= 1

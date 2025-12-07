import pytest

from mindmap_generator import SinhalaMindMapGenerator


@pytest.fixture()
def generator():
    return SinhalaMindMapGenerator()


def test_basic_generation(generator):
    text = "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. එය සුන්දර වෙරළ තීරයන්, පුරාණ නටබුන් සහ පොහොසත් සංස්කෘතියකින් යුක්තය."
    result = generator.generate(text)

    assert result is not None
    assert 'nodes' in result and 'edges' in result and 'metadata' in result
    assert isinstance(result['nodes'], list)
    assert isinstance(result['edges'], list)
    assert result['metadata']['total_nodes'] == len(result['nodes'])


def test_empty_text_returns_empty(generator):
    res = generator.generate("")
    assert isinstance(res, dict)
    assert res['nodes'] == []
    assert res['edges'] == []
    assert res['metadata']['total_nodes'] == 0


def test_root_node_created_once(generator):
    text = "පරිගණකය යනු ඉලෙක්ට්‍රොනික උපකරණයකි. එය ක්‍රියාත්මක වේ."
    res = generator.generate(text)
    root_nodes = [n for n in res['nodes'] if n.get('type') == 'root']
    assert len(root_nodes) == 1
    assert root_nodes[0]['level'] == 0


def test_edges_connect_valid_nodes(generator):
    text = "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. එය සුන්දරයි. සංචාරකයන් පැමිණෙති."
    res = generator.generate(text)
    node_ids = {n['id'] for n in res['nodes']}
    for e in res['edges']:
        assert e['source'] in node_ids
        assert e['target'] in node_ids


def test_metadata_counts_match(generator):
    text = "පරිගණකය උපකරණයකි. එය වැදගත්ය."
    res = generator.generate(text)
    assert res['metadata']['total_nodes'] == len(res['nodes'])
    assert res['metadata']['total_edges'] == len(res['edges'])
    assert res['metadata']['text_length'] == len(text)


def test_limits_on_subtopics_and_details(generator):
    # Create a paragraph with many sentences to exercise limits
    sentences = [f" වාක්‍යය {i} විස්තරයක් ඇත." for i in range(10)]
    long_text = "මෙය ප්‍රධාන මාතෘකාවයි. " + " ".join(sentences)
    res = generator.generate(long_text)

    # Count subtopic nodes (level 2)
    subtopics = [n for n in res['nodes'] if n.get('level') == 2]
    # Implementation limits to 3 subtopics per topic, so expect <= 3 * number_of_topics
    topics = [n for n in res['nodes'] if n.get('level') == 1]
    assert len(subtopics) <= max(3, 3 * max(1, len(topics)))


def test_unique_node_ids(generator):
    text = "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දිවයිනකි. එය සුන්දරයි."
    res = generator.generate(text)
    ids = [n['id'] for n in res['nodes']]
    assert len(ids) == len(set(ids)), "Node ids must be unique"

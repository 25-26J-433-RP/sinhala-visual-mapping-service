from intelligent_mindmap_generator import IntelligentMindMapGenerator
from nlp_engine import SinhalaNLPEngine


def test_known_good_sequence_comparison_and_nested_patterns():
    engine = SinhalaNLPEngine()
    text = (
        "පළමුව දත්ත එකතු කරයි, ඊළඟට විශ්ලේෂණය කරයි, අවසානයේ වාර්තාව සකස් කරයි. "
        "සූර්ය ශක්තිය සහ ගල්අඟුරු අතර වෙනස පැහැදිලිය. "
        "වර්ග: ශාක (මුල්, කඳ, පත්‍ර), සතුන් (ආහාර, වාසස්ථාන)."
    )
    enums = engine.extract_enumerations(text)
    relations = {e.get("relation") for e in enums}

    assert "sequence" in relations
    assert "comparison" in relations
    assert "group" in relations
    group = next((e for e in enums if e.get("relation") == "group"), {})
    assert group.get("nested_items")


def test_known_bad_noise_patterns_do_not_create_large_false_lists():
    engine = SinhalaNLPEngine()
    text = (
        "අද මම පාසලට ගියෙමි සහ මිතුරා හමුවුණා. "
        "මෙය සරල වාක්‍ය දෙකක් පමණි; ලැයිස්තුවක් නොවේ."
    )
    enums = engine.extract_enumerations(text)
    assert len(enums) <= 1


def test_generator_low_confidence_explanations_present_for_weak_inputs():
    generator = IntelligentMindMapGenerator()
    text = "මෙය කෙටි අදහසක් පමණි. සමහර වචන අතර සම්බන්ධය පැහැදිලි නැත."
    result = generator.generate(text)
    explanations = result["metadata"].get("low_confidence_explanations", {})

    assert "summary" in explanations
    assert "low_conf_edges" in explanations["summary"]

from intelligent_mindmap_generator import IntelligentMindMapGenerator
from nlp_engine import SinhalaNLPEngine
from relation_classifier import _FEATURE_NAMES


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


def test_directionality_training_features_present_and_bounded():
    engine = SinhalaNLPEngine()
    classifier = engine.relation_classifier

    assert "cue_position" in _FEATURE_NAMES
    assert "dependency_like_distance" in _FEATURE_NAMES
    assert "clause_boundary" in _FEATURE_NAMES

    pair = {
        "e1": {"text": "වර්ෂාව", "offset": 0},
        "e2": {"text": "ගංවතුර", "offset": 14},
        "context": "වර්ෂාව නිසා, ගංවතුර ඇතිවේ.",
        "same_sentence": True,
        "offset_distance": 14,
        "embed_sim": 0.5,
    }

    feat, _ = classifier._extract_feature_vector(pair)
    index = {name: i for i, name in enumerate(_FEATURE_NAMES)}

    assert feat.shape[0] == len(_FEATURE_NAMES)
    assert 0.0 <= float(feat[index["cue_position"]]) <= 1.0
    assert 0.0 <= float(feat[index["dependency_like_distance"]]) <= 1.0
    assert float(feat[index["clause_boundary"]]) in (0.0, 1.0)
    assert float(feat[index["clause_boundary"]]) == 1.0


def test_related_to_is_rebalanced_against_logical_edges():
    generator = IntelligentMindMapGenerator()

    sample = [
        {
            "source": "A",
            "target": "B",
            "type": "is-a",
            "confidence": 0.74,
            "feature_scores": {"is-a": 0.74, "part-of": 0.08, "cause-effect": 0.05, "related-to": 0.13},
        },
        {
            "source": "C",
            "target": "D",
            "type": "related-to",
            "confidence": 0.96,
            "feature_scores": {"is-a": 0.48, "part-of": 0.10, "cause-effect": 0.09, "related-to": 0.96},
        },
        {
            "source": "E",
            "target": "F",
            "type": "related-to",
            "confidence": 0.92,
            "feature_scores": {"is-a": 0.12, "part-of": 0.09, "cause-effect": 0.08, "related-to": 0.92},
        },
        {
            "source": "G",
            "target": "H",
            "type": "related-to",
            "confidence": 0.89,
            "feature_scores": {"is-a": 0.18, "part-of": 0.11, "cause-effect": 0.10, "related-to": 0.89},
        },
    ]

    rebalanced = generator._rebalance_relationship_edges_for_scoring(sample)
    logical = [r for r in rebalanced if r.get("type") != "related-to"]
    semantic = [r for r in rebalanced if r.get("type") == "related-to"]

    assert logical
    assert semantic
    assert len(semantic) <= len(logical) + 2

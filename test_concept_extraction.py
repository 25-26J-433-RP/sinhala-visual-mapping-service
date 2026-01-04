"""
Comprehensive test and analysis of concept extraction improvements.
"""

from intelligent_mindmap_generator import IntelligentMindMapGenerator
from nlp_engine import SinhalaNLPEngine
import json

def analyze_concept_extraction():
    """Analyze the quality of concept extraction."""
    nlp = SinhalaNLPEngine()
    gen = IntelligentMindMapGenerator()
    
    # Test cases with expected key concepts
    test_cases = [
        {
            'name': 'Photosynthesis',
            'text': '''
            ප්‍රභාසංශ්ලේෂණය ශාක විසින් ආහාර නිපදවන ක්‍රියාවලියයි. 
            ක්ලෝරොප්ලාස්ට් තුළ සිදු වන මෙම ක්‍රියාවලිය සූර්ය ආලෝකය, ජලය සහ කාබන් ඩයොක්සයිඩ් භාවිතා කරයි.
            ග්ලූකෝස් සහ ඔක්සිජන් නිපදවයි. ශාකවල ශක්ති මූලය මෙය ය.
            ''',
            'expected_concepts': ['ප්‍රභාසංශ්ලේෂණය', 'ශාක', 'ක්ලෝරොප්ලාස්ට්', 'ග්ලූකෝස්', 'ඔක්සිජන්', 'සූර්ය ආලෝකය', 'ශක්තി']
        },
        {
            'name': 'Water Cycle',
            'text': '''
            ජල චක්‍රය වාතරණය, ඝනීකරණය සහ අවසාදනය ඇතුළු කරයි.
            සූර්ය තාපය මඟින් ජලය වාතුලයට පරිවර්තනය වේ.
            වාතුවේ ජලය සිසිල් වූ විට ඝනීකරණය ඇතිවේ.
            වස්ත්‍ර සහ මිශ්‍ර ඝනීකරණයෙන් වැසි වැටේ.
            ''',
            'expected_concepts': ['ජල චක්‍රය', 'වාතරණය', 'ඝනීකරණය', 'අවසාදනය', 'වස්ත්‍ර', 'වැසි']
        },
        {
            'name': 'Cell Structure',
            'text': '''
            කෝශ ජීවිතයේ මූල ඒකක වේ.
            핵, සයිටොප්ලാස්ම සහ පටිකය කෝශවල ප්‍රධාන කොටස්ය.
            පෙළ කෝශවල ක්ලෝරොප්ලාස්ට්, මිතෝකොන්ඩ්‍රියා සහ අනෙකුත් ඉංගිතය ඇත.
            සතුවල කෝශවල කෙටි ගුණඉංගිතවතුන් තිබෙයි.
            ''',
            'expected_concepts': ['කෝශ', '핵', 'සයිටොප්ලාස්ම', 'පටිකය', 'ක්ලෝරොප්ලාස්ට්', 'මිතෝකොන්ඩ්‍රියා']
        }
    ]
    
    print("=" * 70)
    print("CONCEPT EXTRACTION ANALYSIS")
    print("=" * 70)
    
    all_results = {}
    
    for test_case in test_cases:
        print(f"\n📚 Test: {test_case['name']}")
        print("-" * 70)
        
        # Extract entities
        entities = nlp.extract_entities(test_case['text'])
        print(f"\nExtracted {len(entities)} entities:")
        for i, entity in enumerate(entities[:8]):
            cleaned = nlp.clean_label(entity['text'])
            print(f"  {i+1}. '{entity['text']}' → '{cleaned}' (importance: {entity['importance']:.2f})")
        
        # Extract key phrases
        key_phrases = nlp.extract_key_phrases(test_case['text'], max_phrases=8)
        print(f"\nExtracted {len(key_phrases)} key phrases:")
        for i, (phrase, score) in enumerate(key_phrases[:8]):
            cleaned = nlp.clean_label(phrase)
            print(f"  {i+1}. '{phrase}' → '{cleaned}' (score: {score:.2f})")
        
        # Generate mind map
        result = gen.generate(test_case['text'], {'max_nodes': 15})
        print(f"\nGenerated mindmap with {len(result['nodes'])} nodes:")
        
        concept_nodes = []
        for node in result['nodes']:
            if node['level'] > 0 and node['label']:  # Exclude root
                concept_nodes.append(node['label'])
                if len(concept_nodes) <= 8:
                    print(f"  {len(concept_nodes)}. [{node['type']}] {node['label']}")
        
        # Check coverage of expected concepts
        all_extracted = ' '.join(concept_nodes).lower()
        covered = []
        missed = []
        
        for concept in test_case['expected_concepts']:
            if concept.lower() in all_extracted:
                covered.append(concept)
            else:
                missed.append(concept)
        
        print(f"\nExpected Concepts Coverage:")
        print(f"  ✓ Covered: {len(covered)}/{len(test_case['expected_concepts'])}")
        if covered:
            for c in covered[:5]:
                print(f"    • {c}")
        if missed:
            print(f"  ✗ Missed: {len(missed)}/{len(test_case['expected_concepts'])}")
            for m in missed[:3]:
                print(f"    • {m}")
        
        all_results[test_case['name']] = {
            'total_nodes': len(result['nodes']),
            'coverage': len(covered),
            'total_expected': len(test_case['expected_concepts']),
            'coverage_rate': len(covered) / len(test_case['expected_concepts']) if test_case['expected_concepts'] else 0
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_coverage_rate = sum(r['coverage_rate'] for r in all_results.values()) / len(all_results)
    print(f"\nOverall Coverage Rate: {total_coverage_rate*100:.1f}%")
    print("\nDetailed Results:")
    for test_name, result in all_results.items():
        rate = result['coverage_rate'] * 100
        print(f"  {test_name}: {result['coverage']}/{result['total_expected']} concepts ({rate:.0f}%)")

def test_phrase_quality():
    """Test the quality of phrase extraction."""
    nlp = SinhalaNLPEngine()
    
    print("\n\n" + "=" * 70)
    print("PHRASE QUALITY ANALYSIS")
    print("=" * 70)
    
    text = '''
    පරිසර විද්‍යාව පৃথිවිය සහ එහි ජීවි ගවේෂණ කරයි.
    ප්‍රාණි හා ශාක එක් සිට එක් අවලම්බනය වේ.
    සෞර ශක්තිය සියලු ජීවන ඉතිරිය මූලය වේ.
    විනාශ සික්ල එ ශක්තිය නැවත බෙදා හරිය.
    '''
    
    phrases = nlp.extract_key_phrases(text, max_phrases=15)
    
    print(f"\nExtracted {len(phrases)} phrases:")
    for i, (phrase, score) in enumerate(phrases):
        cleaned = nlp.clean_label(phrase)
        is_stop_phrase = nlp._is_stop_phrase(phrase)
        status = "⚠ STOP" if is_stop_phrase else "✓ GOOD"
        print(f"  {i+1}. [{status}] '{phrase[:50]}' → '{cleaned[:50]}' (score: {score:.2f})")

if __name__ == '__main__':
    analyze_concept_extraction()
    test_phrase_quality()

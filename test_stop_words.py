"""
Test script to verify stop words are filtered from node labels.
"""

from intelligent_mindmap_generator import IntelligentMindMapGenerator
from mindmap_generator import SinhalaMindMapGenerator

def test_intelligent_generator():
    """Test intelligent generator filters stop words."""
    gen = IntelligentMindMapGenerator()
    
    text = '''
    විද්‍යාව ගැන පාඩම සහ අධ්‍යාපනය.
    ශාක විසින් ආහාර නිපදවයි.
    ක්ලෝරොප්ලාස්ට් තුළ ප්‍රභාසංශ්ලේෂණය සිදු වේ.
    '''
    
    result = gen.generate(text, {'max_nodes': 10})
    
    print("Intelligent Generator Test:")
    print(f"Generated {len(result['nodes'])} nodes")
    
    # Check that stop words are filtered
    stop_words = ['ගැන', 'සහ', 'විසින්', 'තුළ', 'වේ']
    found_stop_words = []
    
    for node in result['nodes']:
        label = node['label']
        for stop_word in stop_words:
            if stop_word in label.split():  # Check as whole word
                found_stop_words.append(f"'{stop_word}' in '{label}'")
    
    if found_stop_words:
        print("❌ FAILED: Found stop words in nodes:")
        for item in found_stop_words:
            print(f"  - {item}")
        return False
    else:
        print("✓ PASSED: No stop words found in node labels")
        print("\nSample nodes (showing concepts only):")
        for i, node in enumerate(result['nodes'][:5]):
            print(f"  {i+1}. {node['label']}")
        return True

def test_regular_generator():
    """Test regular generator filters stop words."""
    gen = SinhalaMindMapGenerator()
    
    text = '''
    විද්‍යාව ගැන පාඩම
    
    ශාක කෝෂ තුළ ක්ලෝරොප්ලාස්ට් ඇත.
    '''
    
    result = gen.generate(text)
    
    print("\n\nRegular Generator Test:")
    print(f"Generated {len(result['nodes'])} nodes")
    
    # Check that stop words are filtered
    stop_words = ['ගැන', 'තුළ']
    found_stop_words = []
    
    for node in result['nodes']:
        label = node['label']
        for stop_word in stop_words:
            if stop_word in label.split():
                found_stop_words.append(f"'{stop_word}' in '{label}'")
    
    if found_stop_words:
        print("❌ FAILED: Found stop words in nodes:")
        for item in found_stop_words:
            print(f"  - {item}")
        return False
    else:
        print("✓ PASSED: No stop words found in node labels")
        print("\nSample nodes (showing concepts only):")
        for i, node in enumerate(result['nodes'][:5]):
            if node['label']:  # Skip empty labels
                print(f"  {i+1}. {node['label']}")
        return True

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Stop Word Filtering in Mind Map Nodes")
    print("=" * 60)
    print()
    
    test1 = test_intelligent_generator()
    test2 = test_regular_generator()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ ALL TESTS PASSED")
        print("Only concepts are displayed in nodes, no helping words!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

#!/usr/bin/env python3
"""Quick test script for intelligent mind map generation."""

from nlp_engine import SinhalaNLPEngine
from intelligent_mindmap_generator import IntelligentMindMapGenerator

print('=' * 60)
print('Testing Intelligent Mind Map Service')
print('=' * 60)

print('\n1. Testing imports...')
engine = SinhalaNLPEngine()
print('   ✓ NLP Engine initialized')

gen = IntelligentMindMapGenerator()
print('   ✓ Generator initialized')

# Test with sample text
print('\n2. Testing with sample Sinhala text...')
sample = "ශ්‍රී ලංකාව දකුණු ආසියාවේ දූපතකි. කොළඹ අගනුවරයි."
result = gen.generate(sample, {'max_nodes': 10})

print(f'   ✓ Generated graph with {result["metadata"]["total_nodes"]} nodes')
print(f'   ✓ Found {result["metadata"]["entities_found"]} entities')
print(f'   ✓ Intelligence level: {result["metadata"]["intelligence_level"]}')

print('\n3. Testing entity extraction...')
entities = engine.extract_entities(sample)
print(f'   ✓ Extracted {len(entities)} entities')
if entities:
    print(f'   Top entity: "{entities[0]["text"]}" (importance: {entities[0]["importance"]:.2f})')

print('\n4. Testing key phrase extraction...')
phrases = engine.extract_key_phrases(sample, max_phrases=3)
print(f'   ✓ Extracted {len(phrases)} key phrases')
if phrases:
    print(f'   Top phrase: "{phrases[0][0]}" (score: {phrases[0][1]:.2f})')

print('\n' + '=' * 60)
print('✅ All components working correctly!')
print('=' * 60)
print('\nReady to use! Start the server with:')
print('  python app.py')
print('\nOr run the example client:')
print('  python example_intelligent_client.py')

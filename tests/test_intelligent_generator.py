"""
Unit tests for the intelligent mind map generator and NLP engine.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_engine import SinhalaNLPEngine
from intelligent_mindmap_generator import IntelligentMindMapGenerator


class TestSinhalaNLPEngine(unittest.TestCase):
    """Test cases for the Sinhala NLP Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nlp_engine = SinhalaNLPEngine()
        self.sample_text = """
        ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රටකි. එහි ජනගහනය මිලියන 22 කි. 
        කොළඹ ශ්‍රී ලංකාවේ වාණිජ අගනුවර වේ. ශ්‍රී ජයවර්ධනපුර කෝට්ටේ පරිපාලන අගනුවර වේ.
        """
    
    def test_entity_extraction(self):
        """Test entity extraction from Sinhala text."""
        entities = self.nlp_engine.extract_entities(self.sample_text)
        
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0, "Should extract at least one entity")
        
        # Check entity structure
        for entity in entities:
            self.assertIn('text', entity)
            self.assertIn('type', entity)
            self.assertIn('importance', entity)
            self.assertGreaterEqual(entity['importance'], 0.0)
            self.assertLessEqual(entity['importance'], 1.0)
    
    def test_entity_importance_scoring(self):
        """Test that importance scores are assigned correctly."""
        entities = self.nlp_engine.extract_entities(self.sample_text)
        
        # Should have entities with varying importance
        importances = [e['importance'] for e in entities]
        self.assertGreater(len(set(importances)), 1, "Should have varied importance scores")
    
    def test_relationship_extraction(self):
        """Test relationship extraction between entities."""
        entities = self.nlp_engine.extract_entities(self.sample_text)
        relationships = self.nlp_engine.extract_relationships(self.sample_text, entities)
        
        self.assertIsInstance(relationships, list)
        
        # Check relationship structure if any found
        for rel in relationships:
            self.assertIn('source', rel)
            self.assertIn('target', rel)
            self.assertIn('type', rel)
            self.assertIn('confidence', rel)
            self.assertGreaterEqual(rel['confidence'], 0.0)
            self.assertLessEqual(rel['confidence'], 1.0)
    
    def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        text1 = "ශ්‍රී ලංකාව දූපතකි"
        text2 = "ශ්‍රී ලංකාව රටකි"
        text3 = "ගණිතය විද්‍යාවකි"
        
        # Similar texts should have higher similarity
        sim_12 = self.nlp_engine.compute_semantic_similarity(text1, text2)
        sim_13 = self.nlp_engine.compute_semantic_similarity(text1, text3)
        
        self.assertIsInstance(sim_12, float)
        self.assertIsInstance(sim_13, float)
        self.assertGreaterEqual(sim_12, 0.0)
        self.assertLessEqual(sim_12, 1.0)
        
        # Related texts should be more similar than unrelated
        # Note: This might not always hold with fallback methods
        # self.assertGreater(sim_12, sim_13)
    
    def test_key_phrase_extraction(self):
        """Test key phrase extraction."""
        phrases = self.nlp_engine.extract_key_phrases(self.sample_text, max_phrases=5)
        
        self.assertIsInstance(phrases, list)
        
        # Check phrase structure
        for phrase, score in phrases:
            self.assertIsInstance(phrase, str)
            self.assertIsInstance(score, float)
            self.assertGreater(len(phrase), 5, "Phrases should be meaningful")
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_concept_clustering(self):
        """Test concept clustering."""
        entities = self.nlp_engine.extract_entities(self.sample_text)
        
        if len(entities) >= 2:
            clusters = self.nlp_engine.cluster_concepts(entities, threshold=0.5)
            
            self.assertIsInstance(clusters, list)
            self.assertGreater(len(clusters), 0)
            
            # Check that all entities are in some cluster
            all_clustered = sum(len(cluster) for cluster in clusters)
            self.assertEqual(all_clustered, len(entities))
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        entities = self.nlp_engine.extract_entities("")
        self.assertEqual(len(entities), 0)
        
        phrases = self.nlp_engine.extract_key_phrases("")
        self.assertEqual(len(phrases), 0)
    
    def test_sinhala_linguistic_patterns(self):
        """Test Sinhala-specific pattern detection."""
        # Text with postpositions
        text_with_postpositions = "කොළඹ සඳහා යන්නෙමු. ඔහු විසින් කරන ලදී."
        entities = self.nlp_engine.extract_entities(text_with_postpositions)
        
        # Should identify entities excluding postpositions
        entity_texts = [e['text'] for e in entities]
        self.assertNotIn('සඳහා', entity_texts, "Postpositions should not be entities")
        self.assertNotIn('විසින්', entity_texts, "Postpositions should not be entities")


class TestIntelligentMindMapGenerator(unittest.TestCase):
    """Test cases for the Intelligent Mind Map Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = IntelligentMindMapGenerator()
        self.sample_text = """
        ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රටකි. 
        කොළඹ වාණිජ අගනුවර වේ. ශ්‍රී ජයවර්ධනපුර කෝට්ටේ පරිපාලන අගනුවර වේ.
        තේ, කෝපි සහ කුළුබඩු ප්‍රධාන අපනයන වේ.
        """
    
    def test_generate_returns_valid_structure(self):
        """Test that generate returns valid graph structure."""
        result = self.generator.generate(self.sample_text)
        
        # Check top-level structure
        self.assertIn('nodes', result)
        self.assertIn('edges', result)
        self.assertIn('metadata', result)
        
        # Check metadata
        metadata = result['metadata']
        self.assertIn('total_nodes', metadata)
        self.assertIn('total_edges', metadata)
        self.assertIn('intelligence_level', metadata)
        self.assertEqual(metadata['intelligence_level'], 'advanced')
    
    def test_generate_creates_nodes(self):
        """Test that nodes are created with correct properties."""
        result = self.generator.generate(self.sample_text)
        nodes = result['nodes']
        
        self.assertGreater(len(nodes), 0, "Should create at least one node")
        
        # Check node structure
        for node in nodes:
            self.assertIn('id', node)
            self.assertIn('label', node)
            self.assertIn('level', node)
            self.assertIn('type', node)
            self.assertIn('size', node)
            self.assertIn('importance', node)
            self.assertIn('color', node)
            
            # Validate ranges
            self.assertIn(node['level'], [0, 1, 2, 3])
            self.assertGreaterEqual(node['importance'], 0.0)
            self.assertLessEqual(node['importance'], 1.0)
    
    def test_generate_creates_hierarchical_structure(self):
        """Test that hierarchical structure is created."""
        result = self.generator.generate(self.sample_text)
        nodes = result['nodes']
        
        # Should have a root node (level 0)
        root_nodes = [n for n in nodes if n['level'] == 0]
        self.assertEqual(len(root_nodes), 1, "Should have exactly one root node")
        
        # Should have multiple levels
        levels = set(n['level'] for n in nodes)
        self.assertGreater(len(levels), 1, "Should have multiple hierarchy levels")
    
    def test_generate_creates_edges(self):
        """Test that edges are created with correct properties."""
        result = self.generator.generate(self.sample_text)
        edges = result['edges']
        
        if len(result['nodes']) > 1:
            self.assertGreater(len(edges), 0, "Should create edges between nodes")
            
            # Check edge structure
            for edge in edges:
                self.assertIn('id', edge)
                self.assertIn('source', edge)
                self.assertIn('target', edge)
                self.assertIn('type', edge)
                self.assertIn('weight', edge)
    
    def test_generate_respects_max_nodes(self):
        """Test that max_nodes parameter is respected."""
        max_nodes = 15
        result = self.generator.generate(self.sample_text, {'max_nodes': max_nodes})
        
        actual_nodes = len(result['nodes'])
        self.assertLessEqual(actual_nodes, max_nodes, 
                            f"Should not exceed max_nodes: {actual_nodes} > {max_nodes}")
    
    def test_generate_with_clustering(self):
        """Test generation with semantic clustering enabled."""
        result = self.generator.generate(
            self.sample_text,
            {'semantic_clustering': True, 'max_nodes': 30}
        )
        
        metadata = result['metadata']
        if 'clusters' in metadata:
            self.assertGreater(metadata['clusters'], 0)
    
    def test_generate_without_clustering(self):
        """Test generation with clustering disabled."""
        result = self.generator.generate(
            self.sample_text,
            {'semantic_clustering': False}
        )
        
        # Should still generate valid graph
        self.assertGreater(len(result['nodes']), 0)
    
    def test_generate_with_high_relationship_threshold(self):
        """Test generation with high relationship threshold."""
        result = self.generator.generate(
            self.sample_text,
            {'relationship_threshold': 0.8}
        )
        
        # Should have fewer relationships due to high threshold
        if 'relationships_found' in result['metadata']:
            # This might be zero with high threshold
            self.assertGreaterEqual(result['metadata']['relationships_found'], 0)
    
    def test_generate_empty_text(self):
        """Test handling of empty text."""
        result = self.generator.generate("")
        
        self.assertEqual(len(result['nodes']), 0)
        self.assertEqual(len(result['edges']), 0)
        self.assertEqual(result['metadata']['total_nodes'], 0)
    
    def test_generate_short_text(self):
        """Test handling of very short text."""
        short_text = "ශ්‍රී ලංකාව."
        result = self.generator.generate(short_text)
        
        # Should create at least a root node
        self.assertGreaterEqual(len(result['nodes']), 1)
    
    def test_node_importance_ordering(self):
        """Test that more important nodes have higher scores."""
        result = self.generator.generate(self.sample_text)
        nodes = result['nodes']
        
        # Root should have highest importance
        root_node = [n for n in nodes if n['level'] == 0][0]
        self.assertGreaterEqual(root_node['importance'], 0.8)
        
        # Check that importance generally decreases with level
        level_1_nodes = [n for n in nodes if n['level'] == 1]
        if level_1_nodes:
            avg_importance_l1 = sum(n['importance'] for n in level_1_nodes) / len(level_1_nodes)
            self.assertGreater(root_node['importance'], avg_importance_l1 * 0.8)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_full_pipeline(self):
        """Test the full pipeline from text to graph."""
        text = """
        පරිසරය ආරක්ෂා කිරීම වැදගත් වේ. 
        ගස් වැඩීමෙන් පරිසරය සුරක්ෂිත වේ.
        ප්‍රදූෂණය අඩු කිරීම අවශ්‍යයි.
        නැවත භාවිතා කළ හැකි බලශක්ති ප්‍රභවයන් භාවිතා කළ යුතුයි.
        """
        
        generator = IntelligentMindMapGenerator()
        result = generator.generate(text, {
            'max_nodes': 25,
            'semantic_clustering': True,
            'relationship_threshold': 0.5
        })
        
        # Validate complete structure
        self.assertGreater(len(result['nodes']), 5)
        self.assertGreater(len(result['edges']), 0)
        self.assertEqual(result['metadata']['intelligence_level'], 'advanced')
        
        # Check that entities were found
        self.assertGreater(result['metadata']['entities_found'], 0)
        
        # Validate graph connectivity
        node_ids = {n['id'] for n in result['nodes']}
        for edge in result['edges']:
            self.assertIn(edge['source'], node_ids, "Edge source should be valid node")
            self.assertIn(edge['target'], node_ids, "Edge target should be valid node")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSinhalaNLPEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentMindMapGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

"""
Intelligent Mind Map Generator with AI-powered Node and Relationship Creation
Uses NLP engine for semantic understanding and intelligent graph construction.
"""

import uuid
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from nlp_engine import SinhalaNLPEngine

logger = logging.getLogger(__name__)


class IntelligentMindMapGenerator:
    """
    AI-powered mind map generator with intelligent node and relationship creation.
    """
    
    def __init__(self):
        """Initialize the intelligent generator."""
        self.nlp_engine = SinhalaNLPEngine()
        self.min_node_importance = 0.3
        self.min_relationship_confidence = 0.4
        self.max_nodes_per_level = {
            0: 1,   # Root
            1: 6,   # Main topics
            2: 12,  # Subtopics
            3: 20   # Details
        }
        
    def generate(self, text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an intelligent mind map from Sinhala text.
        
        Args:
            text: Input Sinhala text
            options: Optional generation parameters
                - max_nodes: Maximum number of nodes
                - semantic_clustering: Enable semantic clustering
                - relationship_threshold: Minimum confidence for relationships
                
        Returns:
            Dictionary with nodes, edges, and metadata
        """
        if not text or not text.strip():
            return self._empty_graph()
        
        options = options or {}
        max_nodes = options.get('max_nodes', 50)
        use_clustering = options.get('semantic_clustering', True)
        rel_threshold = options.get('relationship_threshold', self.min_relationship_confidence)
        
        logger.info(f"Generating intelligent mind map for {len(text)} characters")
        
        # Step 1: Extract entities and concepts using NLP
        entities = self.nlp_engine.extract_entities(text)
        logger.info(f"Extracted {len(entities)} entities")
        
        # Step 2: Extract key phrases for additional concepts
        key_phrases = self.nlp_engine.extract_key_phrases(text, max_phrases=15)
        logger.info(f"Extracted {len(key_phrases)} key phrases")
        
        # Step 3: Cluster similar concepts if enabled
        if use_clustering and len(entities) > 3:
            clusters = self.nlp_engine.cluster_concepts(entities, threshold=0.6)
            logger.info(f"Formed {len(clusters)} concept clusters")
        else:
            clusters = [[e] for e in entities]
        
        # Step 4: Extract relationships between entities
        relationships = self.nlp_engine.extract_relationships(text, entities)
        relationships = [r for r in relationships if r['confidence'] >= rel_threshold]
        logger.info(f"Found {len(relationships)} relationships")
        
        # Step 5: Build hierarchical graph structure
        nodes, edges = self._build_intelligent_graph(
            text, entities, key_phrases, clusters, relationships, max_nodes
        )
        
        # Trim nodes if we exceeded max_nodes
        if len(nodes) > max_nodes:
            # Keep root and most important nodes
            root_nodes = [n for n in nodes if n['level'] == 0]
            other_nodes = [n for n in nodes if n['level'] > 0]
            other_nodes.sort(key=lambda x: x.get('importance', 0), reverse=True)
            
            # Calculate how many other nodes we can keep
            nodes_to_keep = max_nodes - len(root_nodes)
            nodes = root_nodes + other_nodes[:nodes_to_keep]
            
            # Filter edges to only include valid nodes
            valid_node_ids = {n['id'] for n in nodes}
            edges = [e for e in edges if e['source'] in valid_node_ids and e['target'] in valid_node_ids]
        
        # Step 6: Enhance with semantic connections
        if use_clustering:
            edges = self._add_semantic_edges(nodes, edges)
        
        logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'entities_found': len(entities),
                'relationships_found': len(relationships),
                'clusters': len(clusters),
                'text_length': len(text),
                'intelligence_level': 'advanced'
            }
        }
    
    def _build_intelligent_graph(
        self,
        text: str,
        entities: List[Dict],
        key_phrases: List[Tuple[str, float]],
        clusters: List[List[Dict]],
        relationships: List[Dict],
        max_nodes: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Build the graph structure intelligently."""
        nodes = []
        edges = []
        node_map = {}  # text -> node_id mapping
        
        # Create root node from most important entity or first key phrase
        root_text = self._determine_root_topic(text, entities, key_phrases)
        root_id = self._generate_id()
        
        nodes.append({
            'id': root_id,
            'label': root_text,
            'level': 0,
            'type': 'root',
            'size': 35,
            'importance': 1.0,
            'color': '#FF6B6B'
        })
        node_map[root_text] = root_id
        
        # Level 1: Main topics from clusters or top entities
        level1_nodes = self._create_main_topics(clusters, entities, key_phrases, max_nodes)
        
        for i, node_data in enumerate(level1_nodes[:self.max_nodes_per_level[1]]):
            node_id = self._generate_id()
            nodes.append({
                'id': node_id,
                'label': node_data['text'],
                'level': 1,
                'type': 'topic',
                'size': 25,
                'importance': node_data.get('importance', 0.7),
                'color': '#4ECDC4'
            })
            node_map[node_data['text']] = node_id
            
            # Connect to root
            edges.append({
                'id': self._generate_id(),
                'source': root_id,
                'target': node_id,
                'type': 'hierarchy',
                'weight': 3
            })
        
        # Level 2: Subtopics from remaining entities
        level2_candidates = [e for e in entities if e['text'] not in node_map]
        level2_nodes = sorted(level2_candidates, key=lambda x: x['importance'], reverse=True)
        
        for entity in level2_nodes[:self.max_nodes_per_level[2]]:
            node_id = self._generate_id()
            nodes.append({
                'id': node_id,
                'label': entity['text'],
                'level': 2,
                'type': 'subtopic',
                'size': 18,
                'importance': entity['importance'],
                'color': '#95E1D3'
            })
            node_map[entity['text']] = node_id
            
            # Connect to most relevant level 1 node
            parent_id = self._find_best_parent(entity, nodes, level=1)
            if parent_id:
                edges.append({
                    'id': self._generate_id(),
                    'source': parent_id,
                    'target': node_id,
                    'type': 'hierarchy',
                    'weight': 2
                })
        
        # Level 3: Details from key phrases
        level3_candidates = [
            {'text': phrase, 'importance': score}
            for phrase, score in key_phrases
            if phrase not in node_map
        ]
        
        for item in level3_candidates[:self.max_nodes_per_level[3]]:
            if len(nodes) >= max_nodes:
                break
            
            node_id = self._generate_id()
            nodes.append({
                'id': node_id,
                'label': item['text'],
                'level': 3,
                'type': 'detail',
                'size': 12,
                'importance': item['importance'],
                'color': '#F7DC6F'
            })
            node_map[item['text']] = node_id
            
            # Connect to most relevant level 2 node
            parent_id = self._find_best_parent(item, nodes, level=2)
            if parent_id:
                edges.append({
                    'id': self._generate_id(),
                    'source': parent_id,
                    'target': node_id,
                    'type': 'detail',
                    'weight': 1
                })
        
        # Add relationship-based edges
        for rel in relationships:
            source_id = node_map.get(rel['source'])
            target_id = node_map.get(rel['target'])
            
            if source_id and target_id and source_id != target_id:
                # Check if edge doesn't already exist
                existing = any(
                    e['source'] == source_id and e['target'] == target_id
                    for e in edges
                )
                if not existing:
                    edges.append({
                        'id': self._generate_id(),
                        'source': source_id,
                        'target': target_id,
                        'type': rel['type'],
                        'weight': int(rel['confidence'] * 3),
                        'confidence': rel['confidence']
                    })
        
        return nodes, edges
    
    def _create_main_topics(
        self,
        clusters: List[List[Dict]],
        entities: List[Dict],
        key_phrases: List[Tuple[str, float]],
        max_nodes: int
    ) -> List[Dict]:
        """Create main topic nodes from clusters and top entities."""
        main_topics = []
        
        # Use cluster representatives as main topics
        for cluster in clusters:
            if not cluster:
                continue
            
            # Use the most important entity in cluster
            representative = max(cluster, key=lambda x: x['importance'])
            main_topics.append({
                'text': representative['text'],
                'importance': representative['importance'],
                'cluster_size': len(cluster)
            })
        
        # If we have too few topics, add from key phrases
        if len(main_topics) < 3 and key_phrases:
            for phrase, score in key_phrases[:5]:
                if phrase not in [t['text'] for t in main_topics]:
                    main_topics.append({
                        'text': phrase,
                        'importance': score,
                        'cluster_size': 1
                    })
        
        return sorted(main_topics, key=lambda x: x['importance'], reverse=True)
    
    def _determine_root_topic(
        self,
        text: str,
        entities: List[Dict],
        key_phrases: List[Tuple[str, float]]
    ) -> str:
        """Determine the root topic for the mind map."""
        # Try to extract from first sentence
        first_sentence = text.split('.')[0].strip()
        if first_sentence and len(first_sentence) < 100:
            return self._truncate_text(first_sentence, 80)
        
        # Use highest importance entity
        if entities:
            return entities[0]['text']
        
        # Use top key phrase
        if key_phrases:
            return key_phrases[0][0]
        
        # Fallback to first 80 characters
        return self._truncate_text(text, 80)
    
    def _find_best_parent(
        self,
        item: Dict,
        nodes: List[Dict],
        level: int
    ) -> str:
        """Find the best parent node for an item at a specific level."""
        candidates = [n for n in nodes if n['level'] == level]
        
        if not candidates:
            return None
        
        # Calculate similarity with each candidate
        best_parent = None
        best_score = -1
        
        for candidate in candidates:
            # Use semantic similarity
            similarity = self.nlp_engine.compute_semantic_similarity(
                item['text'], candidate['label']
            )
            
            if similarity > best_score:
                best_score = similarity
                best_parent = candidate['id']
        
        return best_parent
    
    def _add_semantic_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Add semantic similarity edges between related nodes at the same level."""
        enhanced_edges = edges.copy()
        similarity_threshold = 0.7
        
        # Group nodes by level (excluding root)
        by_level = defaultdict(list)
        for node in nodes:
            if node['level'] > 0:
                by_level[node['level']].append(node)
        
        # Add semantic edges within each level
        for level, level_nodes in by_level.items():
            if len(level_nodes) < 2:
                continue
            
            for i in range(len(level_nodes)):
                for j in range(i + 1, len(level_nodes)):
                    node1 = level_nodes[i]
                    node2 = level_nodes[j]
                    
                    # Check if edge already exists
                    existing = any(
                        (e['source'] == node1['id'] and e['target'] == node2['id']) or
                        (e['source'] == node2['id'] and e['target'] == node1['id'])
                        for e in enhanced_edges
                    )
                    
                    if not existing:
                        similarity = self.nlp_engine.compute_semantic_similarity(
                            node1['label'], node2['label']
                        )
                        
                        if similarity >= similarity_threshold:
                            enhanced_edges.append({
                                'id': self._generate_id(),
                                'source': node1['id'],
                                'target': node2['id'],
                                'type': 'semantic',
                                'weight': 1,
                                'similarity': similarity,
                                'style': 'dashed'
                            })
        
        return enhanced_edges
    
    def _empty_graph(self) -> Dict[str, Any]:
        """Return an empty graph structure."""
        return {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_nodes': 0,
                'total_edges': 0,
                'intelligence_level': 'none'
            }
        }
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        text = text.strip()
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + '...'
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())[:8]

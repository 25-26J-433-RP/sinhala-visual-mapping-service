"""
Intelligent Mind Map Generator with AI-powered Node and Relationship Creation
Uses NLP engine for semantic understanding and intelligent graph construction.
"""

import uuid
import logging
import math
import hashlib
import re
import unicodedata
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from nlp_engine import SinhalaNLPEngine
from config import Config

logger = logging.getLogger(__name__)


class IntelligentMindMapGenerator:
    """
    AI-powered mind map generator with intelligent node and relationship creation.
    """
    
    def __init__(self):
        """Initialize the intelligent generator."""
        self.nlp_engine = SinhalaNLPEngine()
        self.min_node_importance = getattr(Config, 'MIN_NODE_IMPORTANCE', 0.3)
        self.min_relationship_confidence = getattr(Config, 'MIN_RELATIONSHIP_CONFIDENCE', 0.4)
        self.default_max_nodes = getattr(Config, 'MAX_NODES', 50)
        self.default_semantic_clustering = getattr(Config, 'SEMANTIC_CLUSTERING', True)
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
        if not text or not str(text).strip():
            return self._empty_graph()

        text = self._normalize_text(str(text))
        
        options = options or {}
        max_nodes = options.get('max_nodes', self.default_max_nodes)
        use_clustering = options.get('semantic_clustering', self.default_semantic_clustering)
        rel_threshold = options.get('relationship_threshold', self.min_relationship_confidence)
        
        logger.info(f"Generating intelligent mind map for {len(text)} characters")
        
        # Step 1: Extract entities and concepts using NLP
        entities = self.nlp_engine.extract_entities(text)
        entities = [e for e in entities if e.get('importance', 0) >= self.min_node_importance]
        logger.info(f"Extracted {len(entities)} entities")
        
        # Step 2: Extract key phrases for additional concepts
        key_phrases = self.nlp_engine.extract_key_phrases(text, max_phrases=15)
        logger.info(f"Extracted {len(key_phrases)} key phrases")

        # Step 2b: Merge strong key phrases into entities for richer concepts
        entities = self._merge_entities_with_phrases(text, entities, key_phrases)
        
        # Step 3: Cluster similar concepts if enabled
        if use_clustering and len(entities) > 3:
            clusters = self.nlp_engine.cluster_concepts(entities, threshold=0.6)
            logger.info(f"Formed {len(clusters)} concept clusters")
        else:
            clusters = [[e] for e in entities]

        cluster_map = {}
        for idx, cluster in enumerate(clusters):
            for entity in cluster:
                cluster_map[entity['text']] = idx
        
        # Step 4: Extract relationships between entities
        relationships = self.nlp_engine.extract_relationships(text, entities)
        relationships = [r for r in relationships if r['confidence'] >= rel_threshold]
        logger.info(f"Found {len(relationships)} relationships")

        # Step 4b: Extract structured enumerations (phases, examples, requirements)
        enumerations = self.nlp_engine.extract_enumerations(text)
        logger.info(f"Found {len(enumerations)} enumerations")
        
        # Step 5: Build hierarchical graph structure
        nodes, edges = self._build_intelligent_graph(
            text, entities, key_phrases, clusters, relationships, max_nodes, cluster_map
        )

        # Step 5b: Inject structured enumeration nodes and edges
        nodes, edges = self._apply_enumeration_structure(
            nodes, edges, enumerations, max_nodes
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

        # Step 6b: Add cross-level semantic and proximity edges for richer context
        edges = self._add_cross_level_semantic_edges(nodes, edges)
        edges = self._add_proximity_edges(nodes, edges)

        # Step 7: Assign deterministic positions for better layouts
        nodes = self._assign_positions(nodes)
        
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
        max_nodes: int,
        cluster_map: Dict[str, int]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Build the graph structure intelligently."""
        nodes = []
        edges = []
        node_map = {}  # text -> node_id mapping
        
        # Create root node from most important entity or first key phrase
        root_text = self._determine_root_topic(text, entities, key_phrases)
        root_text = self.nlp_engine.clean_label(root_text)  # Remove helping words
        root_id = self._generate_id()
        root_offset = self._find_offset_in_text(text, root_text)
        
        nodes.append({
            'id': root_id,
            'label': root_text,
            'level': 0,
            'type': 'root',
            'size': 35,
            'importance': 1.0,
            'color': '#FF6B6B',
            'offset': root_offset,
            'source_text': root_text,
            'source_type': 'root'
        })
        node_map[root_text] = root_id
        
        # Level 1: Main topics from clusters or top entities
        level1_nodes = self._create_main_topics(clusters, entities, key_phrases, max_nodes)
        
        for i, node_data in enumerate(level1_nodes[:self.max_nodes_per_level[1]]):
            node_id = self._generate_id()
            cleaned_label = self.nlp_engine.clean_label(node_data['text'])
            nodes.append({
                'id': node_id,
                'label': cleaned_label,
                'level': 1,
                'type': 'topic',
                'size': 25,
                'importance': node_data.get('importance', 0.7),
                'color': '#4ECDC4',
                'offset': self._find_offset_in_text(text, node_data['text']),
                'source_text': node_data['text'],
                'source_type': node_data.get('source_type', 'entity')
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
        level2_nodes = sorted(level2_candidates, key=lambda x: (-(x.get('importance', 0)), x.get('offset', 0)))
        
        for entity in level2_nodes[:self.max_nodes_per_level[2]]:
            node_id = self._generate_id()
            cleaned_label = self.nlp_engine.clean_label(entity['text'])
            nodes.append({
                'id': node_id,
                'label': cleaned_label,
                'level': 2,
                'type': 'subtopic',
                'size': 18,
                'importance': entity['importance'],
                'color': '#95E1D3',
                'offset': entity.get('offset', self._find_offset_in_text(text, entity['text'])),
                'source_text': entity['text'],
                'source_type': entity.get('type', 'entity')
            })
            node_map[entity['text']] = node_id
            
            # Connect to most relevant level 1 node
            parent_id = self._find_best_parent(entity, nodes, level=1, cluster_map=cluster_map)
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
            cleaned_label = self.nlp_engine.clean_label(item['text'])
            nodes.append({
                'id': node_id,
                'label': cleaned_label,
                'level': 3,
                'type': 'detail',
                'size': 12,
                'importance': item['importance'],
                'color': '#F7DC6F',
                'offset': self._find_offset_in_text(text, item['text']),
                'source_text': item['text'],
                'source_type': 'phrase'
            })
            node_map[item['text']] = node_id
            
            # Connect to most relevant level 2 node
            parent_id = self._find_best_parent(item, nodes, level=2, cluster_map=cluster_map)
            if parent_id:
                edges.append({
                    'id': self._generate_id(),
                    'source': parent_id,
                    'target': node_id,
                    'type': 'detail',
                    'weight': 1
                })
        
        # Add relationship-based edges with deduplication (undirected canonical key)
        edge_keys = set()
        for rel in relationships:
            source_id = node_map.get(rel['source'])
            target_id = node_map.get(rel['target'])
            
            if source_id and target_id and source_id != target_id:
                key = tuple(sorted([source_id, target_id]) + [rel['type']])
                if key in edge_keys:
                    continue
                edge_keys.add(key)
                edges.append({
                    'id': self._generate_id(),
                    'source': source_id,
                    'target': target_id,
                    'type': rel['type'],
                    'weight': max(1, int(rel['confidence'] * 3)),
                    'confidence': rel['confidence']
                })
        
        return nodes, edges

    def _apply_enumeration_structure(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        enumerations: List[Dict],
        max_nodes: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Add structured list-based nodes/edges like phases and examples."""
        if not enumerations:
            return nodes, edges

        node_lookup = {n['label']: n['id'] for n in nodes}
        root_id = next((n['id'] for n in nodes if n.get('level') == 0), None)

        def ensure_node(label: str, level: int, node_type: str, size: int, source_type: str) -> str:
            if label in node_lookup:
                return node_lookup[label]
            if len(nodes) >= max_nodes:
                return None
            node_id = self._generate_id()
            nodes.append({
                'id': node_id,
                'label': label,
                'level': level,
                'type': node_type,
                'size': size,
                'importance': 0.6,
                'color': '#C7C7FF',
                'offset': self._find_offset_in_text(' '.join([label]), label),
                'source_text': label,
                'source_type': source_type
            })
            node_lookup[label] = node_id
            return node_id

        for enum in enumerations:
            head_label = self.nlp_engine.clean_label(enum.get('head') or '') if enum.get('head') else None
            items = [self.nlp_engine.clean_label(i) for i in enum.get('items', [])]
            items = [i for i in items if i]
            if len(items) < 2:
                continue

            relation = enum.get('relation')
            head_id = None

            if head_label:
                head_id = ensure_node(head_label, 1, 'topic', 22, 'enumeration_head')
                if head_id and root_id:
                    edges.append({
                        'id': self._generate_id(),
                        'source': root_id,
                        'target': head_id,
                        'type': 'includes',
                        'weight': 2
                    })

            # If no head, attach to root
            if not head_id and root_id:
                head_id = root_id

            previous_id = None
            for item in items:
                level = 2 if head_id != root_id else 1
                item_id = ensure_node(item, level, 'subtopic', 16, 'enumeration_item')
                if not item_id:
                    continue

                edge_type = 'example_of' if relation == 'example' else 'phase'
                if relation == 'requires':
                    edge_type = 'requires'
                edges.append({
                    'id': self._generate_id(),
                    'source': head_id,
                    'target': item_id,
                    'type': edge_type,
                    'weight': 2
                })

                if relation == 'sequence' and previous_id:
                    edges.append({
                        'id': self._generate_id(),
                        'source': previous_id,
                        'target': item_id,
                        'type': 'followed_by',
                        'weight': 1
                    })
                previous_id = item_id

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
            representative = self._select_cluster_representative(cluster)
            main_topics.append({
                'text': representative['text'],
                'importance': self._score_cluster(cluster),
                'cluster_size': len(cluster),
                'source_type': representative.get('type', 'entity')
            })
        
        # If we have too few topics, add from key phrases
        if len(main_topics) < 3 and key_phrases:
            for phrase, score in key_phrases[:5]:
                if phrase not in [t['text'] for t in main_topics]:
                    main_topics.append({
                        'text': phrase,
                        'importance': score,
                        'cluster_size': 1,
                        'source_type': 'phrase'
                    })
        
        return sorted(main_topics, key=lambda x: x['importance'], reverse=True)
    
    def _determine_root_topic(
        self,
        text: str,
        entities: List[Dict],
        key_phrases: List[Tuple[str, float]]
    ) -> str:
        """Determine the root topic for the mind map."""
        # Prefer strong entities near the beginning of the text
        if entities:
            text_len = max(1, len(text))
            early_entities = [
                e for e in entities
                if e.get('offset', text_len) <= int(text_len * 0.35)
            ]
            if early_entities:
                early_entities.sort(key=lambda x: (-(x.get('importance', 0)), x.get('offset', 0)))
                return early_entities[0]['text']
            entities.sort(key=lambda x: (-(x.get('importance', 0)), x.get('offset', 0)))
            return entities[0]['text']

        # Try to extract from first sentence
        first_sentence = text.split('.')[0].strip()
        if first_sentence and len(first_sentence) < 100:
            return self._truncate_text(first_sentence, 80)
        
        # Use top key phrase
        if key_phrases:
            return key_phrases[0][0]
        
        # Fallback to first 80 characters
        return self._truncate_text(text, 80)
    
    def _find_best_parent(
        self,
        item: Dict,
        nodes: List[Dict],
        level: int,
        cluster_map: Dict[str, int] = None
    ) -> str:
        """Find the best parent node for an item at a specific level."""
        candidates = [n for n in nodes if n['level'] == level]
        
        if not candidates:
            return None
        
        # Calculate similarity with each candidate
        best_parent = None
        best_score = -1
        
        item_text = item['text']
        item_offset = item.get('offset')

        for candidate in candidates:
            candidate_text = candidate.get('source_text', candidate['label'])
            similarity = self.nlp_engine.compute_semantic_similarity(
                item_text, candidate['label']
            )

            proximity_score = 0.0
            candidate_offset = candidate.get('offset')
            if item_offset is not None and candidate_offset is not None:
                distance = abs(item_offset - candidate_offset)
                proximity_score = max(0.0, 1.0 - min(distance, 400) / 400.0)

            cluster_bonus = 0.0
            if cluster_map:
                item_cluster = cluster_map.get(item_text)
                candidate_cluster = cluster_map.get(candidate_text)
                if item_cluster is not None and item_cluster == candidate_cluster:
                    cluster_bonus = 1.0

            score = (0.6 * similarity) + (0.25 * proximity_score) + (0.15 * cluster_bonus)

            if score > best_score:
                best_score = score
                best_parent = candidate['id']
        
        return best_parent

    def _select_cluster_representative(self, cluster: List[Dict]) -> Dict:
        """Pick a representative entity that is central and important."""
        if not cluster:
            return {'text': '', 'importance': 0.0}
        if len(cluster) == 1:
            return cluster[0]

        best_item = None
        best_score = -1.0
        for item in cluster:
            centrality = 0.0
            for other in cluster:
                if item is other:
                    continue
                centrality += self.nlp_engine.compute_semantic_similarity(
                    item['text'], other['text']
                )
            score = item.get('importance', 0) + (centrality / max(1, len(cluster) - 1))
            if score > best_score:
                best_score = score
                best_item = item
        return best_item or cluster[0]

    def _score_cluster(self, cluster: List[Dict]) -> float:
        """Score cluster importance using size and average entity importance."""
        if not cluster:
            return 0.0
        avg_importance = sum(e.get('importance', 0) for e in cluster) / len(cluster)
        size_boost = min(0.3, math.log(len(cluster) + 1, 3) * 0.2)
        return min(1.0, avg_importance + size_boost)

    def _merge_entities_with_phrases(
        self,
        text: str,
        entities: List[Dict],
        key_phrases: List[Tuple[str, float]]
    ) -> List[Dict]:
        """Promote high-quality key phrases into entities and dedupe by normalized form."""
        if not key_phrases:
            return entities

        normalized_map = {}
        for entity in entities:
            key = entity.get('normalized') or self.nlp_engine._normalize_term(entity['text'])
            normalized_map[key] = entity

        for phrase, score in key_phrases:
            if score < max(self.min_node_importance, 0.35):
                continue
            norm = self.nlp_engine._normalize_term(phrase)
            if not norm or norm in normalized_map:
                continue

            normalized_map[norm] = {
                'text': phrase,
                'type': 'concept_phrase',
                'importance': min(1.0, score * 1.1),
                'context': 'phrase_extraction',
                'offset': self._find_offset_in_text(text, phrase),
                'normalized': norm,
                'frequency': text.count(phrase)
            }

        return list(normalized_map.values())
    
    def _add_semantic_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Add semantic similarity edges between related nodes at the same level."""
        enhanced_edges = edges.copy()
        similarity_threshold = 0.75
        max_edges_per_node = 2
        
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
                node1 = level_nodes[i]
                candidates = []
                for j in range(len(level_nodes)):
                    if i == j:
                        continue
                    node2 = level_nodes[j]

                    existing = any(
                        (e['source'] == node1['id'] and e['target'] == node2['id']) or
                        (e['source'] == node2['id'] and e['target'] == node1['id'])
                        for e in enhanced_edges
                    )
                    if existing:
                        continue

                    similarity = self.nlp_engine.compute_semantic_similarity(
                        node1['label'], node2['label']
                    )
                    if similarity >= similarity_threshold:
                        candidates.append((similarity, node2))

                candidates.sort(key=lambda x: x[0], reverse=True)
                for similarity, node2 in candidates[:max_edges_per_node]:
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

    def _add_cross_level_semantic_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Connect semantically close nodes across adjacent levels for richer structure."""
        enhanced = edges.copy()
        similarity_threshold = 0.82
        max_edges_per_parent = 2
        # Index nodes by level for quick lookup
        by_level = defaultdict(list)
        for node in nodes:
            by_level[node['level']].append(node)

        for level in range(0, 3):
            lower = by_level.get(level)
            upper = by_level.get(level + 1)
            if not lower or not upper:
                continue

            for parent in lower:
                candidates = []
                for child in upper:
                    # Skip if already connected
                    exists = any(
                        (e['source'] == parent['id'] and e['target'] == child['id']) or
                        (e['source'] == child['id'] and e['target'] == parent['id'])
                        for e in enhanced
                    )
                    if exists:
                        continue

                    similarity = self.nlp_engine.compute_semantic_similarity(parent['label'], child['label'])
                    if similarity >= similarity_threshold:
                        candidates.append((similarity, child))

                candidates.sort(key=lambda x: x[0], reverse=True)
                for similarity, child in candidates[:max_edges_per_parent]:
                    enhanced.append({
                        'id': self._generate_id(),
                        'source': parent['id'],
                        'target': child['id'],
                        'type': 'semantic_cross',
                        'weight': 1,
                        'similarity': similarity,
                        'style': 'dotted'
                    })
        return enhanced

    def _add_proximity_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Link nearby concepts based on text order to surface narrative flow."""
        enhanced = edges.copy()
        sorted_nodes = [n for n in nodes if 'offset' in n]
        sorted_nodes.sort(key=lambda n: n.get('offset', 10**9))

        for i, node in enumerate(sorted_nodes[:-1]):
            neighbor = sorted_nodes[i + 1]

            # Skip if already connected
            exists = any(
                (e['source'] == node['id'] and e['target'] == neighbor['id']) or
                (e['source'] == neighbor['id'] and e['target'] == node['id'])
                for e in enhanced
            )
            if exists:
                continue

            distance = abs(node.get('offset', 0) - neighbor.get('offset', 0))
            if distance > 400:
                continue

            enhanced.append({
                'id': self._generate_id(),
                'source': node['id'],
                'target': neighbor['id'],
                'type': 'context',
                'weight': 1,
                'proximity': distance,
                'style': 'dashed'
            })

        return enhanced

    def _assign_positions(self, nodes: List[Dict]) -> List[Dict]:
        """Assign deterministic positions for nodes to make layouts stable per essay."""
        if not nodes:
            return nodes

        # Group by level and sort deterministically by importance then text hash
        levels = defaultdict(list)
        for node in nodes:
            levels[node['level']].append(node)

        for level, level_nodes in levels.items():
            # Preserve reading order by offset, then importance
            level_nodes.sort(key=lambda n: (n.get('offset', 10**9), -n.get('importance', 0), self._stable_hash(n['label'])))
            count = len(level_nodes)
            base_radius = 170 * (level + 1)
            golden_angle = math.pi * (3 - math.sqrt(5))  # ~2.399963

            for idx, node in enumerate(level_nodes):
                radius_scale = 1 + 0.12 * (count / 8)
                radius = base_radius * radius_scale
                angle = golden_angle * idx + 0.18 * level

                # Add tiny deterministic jitter to avoid perfect overlaps
                jitter = (self._stable_hash(node['id']) % 7) / 100.0
                x = radius * math.cos(angle) * (1 + jitter)
                y = radius * math.sin(angle) * (1 + jitter)

                node['position'] = {
                    'x': round(x, 2),
                    'y': round(y, 2)
                }
                node['order'] = idx

        return nodes

    def _stable_hash(self, text: str) -> int:
        """Deterministic hash for layout ordering."""
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)

    def _find_offset_in_text(self, text: str, phrase: str) -> int:
        """Find the first occurrence offset of phrase within text."""
        try:
            match = re.search(re.escape(phrase), text)
            if match:
                return match.start()
        except Exception:
            pass
        return 0
    
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

    def _normalize_text(self, text: str) -> str:
        """Normalize control characters while preserving Sinhala-friendly spacing."""
        try:
            text = unicodedata.normalize('NFC', text)
        except Exception:
            pass
        cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
        cleaned = re.sub(r'[ \t\r\f\v]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

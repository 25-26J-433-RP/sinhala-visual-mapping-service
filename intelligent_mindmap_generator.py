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
from graph_constraints import GraphConstraints

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
        self.graph_constraints = GraphConstraints()
        self.base_nodes_per_level = {
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
        all_entities = self.nlp_engine.extract_entities(text)
        entities = [e for e in all_entities if e.get('importance', 0) >= self.min_node_importance]
        if len(entities) < 6 and len(all_entities) > len(entities):
            relaxed_target = max(6, min(14, len(all_entities)))
            entities = all_entities[:relaxed_target]
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
        raw_relationships = self.nlp_engine.extract_relationships(text, entities)
        relationships = [r for r in raw_relationships if r['confidence'] >= rel_threshold]
        if not relationships and raw_relationships:
            relaxed_rel_threshold = max(0.26, rel_threshold - 0.15)
            relationships = [
                r for r in raw_relationships if r.get('confidence', 0.0) >= relaxed_rel_threshold
            ]
        logger.info(f"Found {len(relationships)} relationships")

        # Step 4b: Extract structured enumerations (phases, examples, requirements)
        enumerations = self.nlp_engine.extract_enumerations(text)
        logger.info(f"Found {len(enumerations)} enumerations")

        # Step 4c: Dynamic node limits based on essay complexity
        node_limits = self._compute_dynamic_node_limits(
            text=text,
            entities=entities,
            key_phrases=key_phrases,
            max_nodes=max_nodes,
        )
        logger.info(f"Dynamic node limits: {node_limits}")
        
        # Step 5: Build hierarchical graph structure
        nodes, edges = self._build_intelligent_graph(
            text, entities, key_phrases, clusters, relationships, max_nodes, cluster_map, node_limits
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

        # Step 6c: CRITICAL - Deduplicate edges to show only one line between any two nodes
        edges = self._deduplicate_edges(edges)

        # Step 6d: Graph constraint post-processing
        #   1. Merge near-duplicate nodes
        #   2. Remove cycles in hierarchy edges
        #   3. Limit parents per node (max 2 hierarchy parents)
        #   4. Drop weak cross-cluster soft edges
        nodes, edges = self.graph_constraints.apply(nodes, edges, cluster_map)

        # Step 6e: Coherence-aware pruning for soft cross-level spaghetti edges
        edges = self._prune_spaghetti_edges(nodes, edges)

        # Step 7: Assign deterministic positions for better layouts
        nodes = self._assign_positions(nodes)

        coherence_metrics = self._compute_graph_coherence(nodes, edges)
        sentence_count = max(1, len(self.nlp_engine._split_sentences_with_spans(text)))
        quality_metrics = self._compute_quality_metrics(text, nodes, edges, coherence_metrics, sentence_count)
        confidence_distribution = self._compute_confidence_distribution(nodes, edges)
        low_confidence_explanations = self._build_low_confidence_explanations(nodes, edges)
        
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
                'coherence': coherence_metrics,
                'quality': quality_metrics,
                'confidence_distribution': confidence_distribution,
                'semantic_density': quality_metrics.get('semantic_density', 0.0),
                'semantic_coverage': quality_metrics.get('semantic_coverage', 0.0),
                'redundancy': quality_metrics.get('redundancy', 0.0),
                'low_confidence_explanations': low_confidence_explanations,
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
        cluster_map: Dict[str, int],
        node_limits: Dict[int, int]
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
        
        # Branch colors for visual distinction
        branch_colors = ['#4ECDC4', '#95E1D3', '#F38181', '#FFA07A', '#98D8C8']
        
        for i, node_data in enumerate(level1_nodes[:node_limits[1]]):
            node_id = self._generate_id()
            # Keep label clean but preserve meaning
            cleaned_label = self._clean_label_preserving(node_data['text'])
            nodes.append({
                'id': node_id,
                'label': cleaned_label,
                'level': 1,
                'type': 'topic',
                'size': 25,
                'importance': node_data.get('importance', 0.7),
                'color': branch_colors[i % len(branch_colors)],
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
        
        for entity in level2_nodes[:node_limits[2]]:
            node_id = self._generate_id()
            # Preserve label integrity for subtopics
            cleaned_label = self._clean_label_preserving(entity['text'])
            
            # Match parent's color for branch consistency
            parent_id = self._find_best_parent(entity, nodes, level=1, cluster_map=cluster_map)
            parent_color = '#95E1D3'
            if parent_id:
                parent_node = next((n for n in nodes if n['id'] == parent_id), None)
                if parent_node:
                    parent_color = parent_node['color']
            
            nodes.append({
                'id': node_id,
                'label': cleaned_label,
                'level': 2,
                'type': 'subtopic',
                'size': 18,
                'importance': entity['importance'],
                'color': parent_color,
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
        
        for item in level3_candidates[:node_limits[3]]:
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
        
        # Add relationship-based edges.
        # Deduplication key is DIRECTED (source, target, type) so the
        # correct orientation from the relation classifier is preserved.
        # When the reverse orientation of the same type was already seen
        # we keep whichever has the higher direction_score.
        scored_relationships = self._rebalance_relationship_edges_for_scoring(relationships)
        edge_keys: Dict[tuple, Dict] = {}
        for rel in scored_relationships:
            source_id = node_map.get(rel['source'])
            target_id = node_map.get(rel['target'])

            if source_id and target_id and source_id != target_id:
                fwd_key = (source_id, target_id, rel['type'])
                rev_key = (target_id, source_id, rel['type'])
                new_dir = rel.get('direction_score', 0.5)
                rel_conf = float(rel.get('confidence', 0.0))
                is_semantic = rel.get('type') == 'related-to'
                edge_score = rel_conf * (0.82 if is_semantic else 1.0)
                edge_weight = max(1, int((edge_score if not is_semantic else edge_score * 0.9) * 3))
                edge_dict = {
                    'id': self._generate_id(),
                    'source': source_id,
                    'target': target_id,
                    'type': rel['type'],
                    'weight': edge_weight,
                    'confidence': rel_conf,
                    'edge_score': round(edge_score, 4),
                    'relation_family': rel.get(
                        'relation_family',
                        'semantic_relatedness' if is_semantic else 'logical_relation'
                    ),
                    'direction_score': new_dir,
                    'directed': rel.get('directed', False),
                }
                if rev_key in edge_keys:
                    # Both directions found; keep the one with stronger cue
                    if new_dir > edge_keys[rev_key].get('direction_score', 0.5):
                        del edge_keys[rev_key]
                        edge_keys[fwd_key] = edge_dict
                elif fwd_key not in edge_keys or \
                    edge_score > edge_keys[fwd_key].get('edge_score', edge_keys[fwd_key].get('confidence', 0.0)):
                    edge_keys[fwd_key] = edge_dict

        edges.extend(edge_keys.values())
        
        return nodes, edges

    def _rebalance_relationship_edges_for_scoring(self, relationships: List[Dict]) -> List[Dict]:
        """
        Separate logical relations from semantic relatedness and cap soft
        semantic edges so they do not dominate graph connectivity.
        """
        if not relationships:
            return []

        logical = sorted(
            [r for r in relationships if r.get('type') != 'related-to'],
            key=lambda r: float(r.get('confidence', 0.0)),
            reverse=True,
        )
        semantic = sorted(
            [r for r in relationships if r.get('type') == 'related-to'],
            key=lambda r: float(r.get('confidence', 0.0)),
            reverse=True,
        )

        if not semantic:
            return logical

        if not logical:
            semantic_cap = min(len(semantic), 6)
        else:
            semantic_cap = min(len(semantic), len(logical) + 2)

        selected = logical + semantic[:semantic_cap]
        selected.sort(key=lambda r: float(r.get('confidence', 0.0)), reverse=True)
        return selected

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
            nested_items = enum.get('nested_items', {}) or {}
            enum_confidence = float(enum.get('confidence', 0.58))
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
                elif relation == 'comparison':
                    edge_type = 'compares'
                elif relation == 'argument':
                    edge_type = 'argues'
                elif relation == 'group':
                    edge_type = 'includes'
                edges.append({
                    'id': self._generate_id(),
                    'source': head_id,
                    'target': item_id,
                    'type': edge_type,
                    'weight': 2,
                    'confidence': enum_confidence,
                })

                child_items = [self.nlp_engine.clean_label(c) for c in nested_items.get(item, [])]
                child_items = [c for c in child_items if c]
                for child in child_items:
                    child_id = ensure_node(child, min(level + 1, 3), 'detail', 14, 'enumeration_nested_item')
                    if not child_id:
                        continue
                    edges.append({
                        'id': self._generate_id(),
                        'source': item_id,
                        'target': child_id,
                        'type': 'subitem_of',
                        'weight': 1,
                        'confidence': max(0.42, enum_confidence - 0.08),
                    })

                if relation == 'sequence' and previous_id:
                    edges.append({
                        'id': self._generate_id(),
                        'source': previous_id,
                        'target': item_id,
                        'type': 'followed_by',
                        'weight': 1,
                        'confidence': max(0.45, enum_confidence - 0.05),
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
        """Determine root topic using topic-sentence and subject salience scoring."""
        if entities:
            sentences = self.nlp_engine._split_sentences_with_spans(text)
            topic_candidates = self._detect_topic_sentence_entities(sentences, entities)
            if topic_candidates:
                return topic_candidates[0]['text']

            # Fallback: pick globally salient main subject, not just first mention
            scored = []
            text_len = max(1, len(text))
            for e in entities:
                importance = float(e.get('importance', 0.0))
                offset = float(e.get('offset', text_len))
                position_score = max(0.0, 1.0 - (offset / text_len))
                type_bonus = 0.15 if e.get('type') in {'concept', 'proper_noun', 'location'} else 0.0
                freq_bonus = min(0.2, (text.count(e.get('text', '')) - 1) * 0.05)
                salience = (0.55 * importance) + (0.25 * position_score) + type_bonus + freq_bonus
                scored.append((salience, e))

            scored.sort(key=lambda x: x[0], reverse=True)
            if scored:
                return scored[0][1]['text']

        if key_phrases:
            return key_phrases[0][0]

        first_sentence = (text.split('.')[0] if '.' in text else text[:120]).strip()
        if first_sentence:
            words = first_sentence.split()
            return ' '.join(words[:2]) if len(words) >= 2 else self._truncate_text(first_sentence, 80)
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
        # Stricter threshold for cleaner visualization
        similarity_threshold = 0.85
        max_edges_per_node = 1  # Reduced from 2
        
        # Group nodes by level (excluding root)
        by_level = defaultdict(list)
        for node in nodes:
            if node['level'] > 0:
                by_level[node['level']].append(node)
        
        # Add semantic edges within each level (only for level 2+ to reduce clutter)
        for level, level_nodes in by_level.items():
            if len(level_nodes) < 2 or level < 2:  # Skip level 1
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
        # DISABLED: Too many cross-level edges create visual clutter
        # For cleaner mind maps, rely on hierarchical edges only
        return edges

    def _add_proximity_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Link nearby concepts based on text order to surface narrative flow."""
        # DISABLED: Proximity edges create too much clutter
        # For cleaner visualization, use hierarchy only
        return edges

    def _deduplicate_edges(self, edges: List[Dict]) -> List[Dict]:
        """
        Deduplicate exact duplicate edges while preserving valid parallel relations.

        Keeps one edge per (source, target, type) signature. This avoids
        over-deduplication that previously removed meaningful parallel edges
        between the same two nodes with different relation types.
        """
        if not edges:
            return edges

        grouped: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
        for edge in edges:
            key = (edge.get('source'), edge.get('target'), edge.get('type', ''))
            grouped[key].append(edge)

        deduplicated: List[Dict] = []
        for same_edges in grouped.values():
            if len(same_edges) == 1:
                deduplicated.append(same_edges[0])
                continue
            best = max(
                same_edges,
                key=lambda e: (
                    e.get('weight', 1),
                    e.get('confidence', 0.5),
                    e.get('direction_score', 0.5),
                ),
            )
            deduplicated.append(best)

        return deduplicated

    def _compute_dynamic_node_limits(
        self,
        text: str,
        entities: List[Dict],
        key_phrases: List[Tuple[str, float]],
        max_nodes: int,
    ) -> Dict[int, int]:
        """Set level-wise node limits dynamically from essay complexity."""
        text_len = len(text)
        sentence_count = max(1, len(self.nlp_engine._split_sentences_with_spans(text)))
        entity_density = len(entities) / sentence_count
        phrase_density = len(key_phrases) / sentence_count

        complexity = 0.0
        complexity += min(1.0, text_len / 1600.0) * 0.40
        complexity += min(1.0, entity_density / 4.0) * 0.35
        complexity += min(1.0, phrase_density / 3.0) * 0.25

        budget = max(12, min(max_nodes, int(max_nodes * (0.60 + 0.40 * complexity))))
        level1 = max(4, int(budget * 0.22))
        level2 = max(6, int(budget * 0.40))
        level3 = max(6, budget - 1 - level1 - level2)

        return {
            0: 1,
            1: min(level1, 12),
            2: min(level2, 24),
            3: min(level3, 32),
        }

    def _detect_topic_sentence_entities(
        self,
        sentences: List[Tuple[str, int]],
        entities: List[Dict],
    ) -> List[Dict]:
        """Return likely main-subject entities from topic sentences."""
        if not sentences or not entities:
            return []

        candidate_spans = sentences[: min(3, len(sentences))]
        topic_markers = ('යනු', 'වන්නේ', 'මෙම', 'මෙය', 'අර්ථයෙන්', 'සඳහා', 'විෂය')
        selected: List[Dict] = []

        for sent_text, sent_start in candidate_spans:
            sent_end = sent_start + len(sent_text)
            sentence_bonus = 0.15 if any(m in sent_text for m in topic_markers) else 0.0

            in_sentence = [
                e for e in entities
                if sent_start <= e.get('offset', -1) < sent_end
            ]
            for e in in_sentence:
                importance = float(e.get('importance', 0.0))
                offset_local = max(0, e.get('offset', sent_start) - sent_start)
                locality = max(0.0, 1.0 - (offset_local / max(1, len(sent_text))))
                score = (0.65 * importance) + (0.20 * locality) + sentence_bonus
                selected.append({**e, '_topic_score': score})

        selected.sort(key=lambda x: x.get('_topic_score', 0.0), reverse=True)
        return selected

    def _prune_spaghetti_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Aggressively prune weak soft cross-level edges and high-degree clutter."""
        if not edges:
            return edges

        node_level = {n['id']: n.get('level', 0) for n in nodes}
        soft_types = {'semantic', 'semantic_cross', 'proximity', 'related-to', 'conjunction', 'close'}

        filtered: List[Dict] = []
        soft_out_degree: Dict[str, int] = defaultdict(int)

        for e in edges:
            etype = e.get('type', '')
            src, tgt = e.get('source'), e.get('target')
            l_src = node_level.get(src, 0)
            l_tgt = node_level.get(tgt, 0)
            level_gap = abs(l_src - l_tgt)
            conf = float(e.get('confidence', 0.5))
            weight = float(e.get('weight', 1))

            if etype in soft_types:
                if level_gap >= 2 and conf < 0.72:
                    continue
                if level_gap >= 1 and conf < 0.62 and weight <= 1:
                    continue
                if soft_out_degree[src] >= 3:
                    continue
                soft_out_degree[src] += 1

            filtered.append(e)

        return filtered

    def _compute_graph_coherence(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, float]:
        """Compute graph quality metrics for connectivity and depth balance."""
        if not nodes:
            return {
                'connectivity': 0.0,
                'depth_balance': 0.0,
                'cross_level_ratio': 0.0,
                'avg_degree': 0.0,
                'coherence_score': 0.0,
            }

        node_ids = [n['id'] for n in nodes]
        id_set = set(node_ids)
        adj: Dict[str, List[str]] = defaultdict(list)
        degree: Dict[str, int] = defaultdict(int)
        cross_level = 0

        levels = {n['id']: n.get('level', 0) for n in nodes}
        for e in edges:
            s, t = e.get('source'), e.get('target')
            if s not in id_set or t not in id_set:
                continue
            adj[s].append(t)
            adj[t].append(s)
            degree[s] += 1
            degree[t] += 1
            if levels.get(s, 0) != levels.get(t, 0):
                cross_level += 1

        # Connectivity: fraction of nodes reachable from root (or first node)
        root_id = next((n['id'] for n in nodes if n.get('level') == 0), node_ids[0])
        visited = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            stack.extend(adj.get(cur, []))
        connectivity = len(visited) / max(1, len(nodes))

        # Depth balance: reward spread across levels without extreme skew
        level_counts: Dict[int, int] = defaultdict(int)
        for n in nodes:
            level_counts[n.get('level', 0)] += 1
        counts = list(level_counts.values())
        mean_count = sum(counts) / max(1, len(counts))
        variance = sum((c - mean_count) ** 2 for c in counts) / max(1, len(counts))
        depth_balance = 1.0 / (1.0 + variance / max(1.0, mean_count))

        cross_level_ratio = cross_level / max(1, len(edges))
        avg_degree = sum(degree.values()) / max(1, len(nodes))
        coherence_score = (
            (0.45 * connectivity)
            + (0.30 * depth_balance)
            + (0.15 * max(0.0, 1.0 - cross_level_ratio))
            + (0.10 * min(1.0, avg_degree / 2.5))
        )

        return {
            'connectivity': round(float(connectivity), 4),
            'depth_balance': round(float(depth_balance), 4),
            'cross_level_ratio': round(float(cross_level_ratio), 4),
            'avg_degree': round(float(avg_degree), 4),
            'coherence_score': round(float(coherence_score), 4),
        }

    def _compute_quality_metrics(
        self,
        text: str,
        nodes: List[Dict],
        edges: List[Dict],
        coherence_metrics: Dict[str, float],
        sentence_count: int,
    ) -> Dict[str, float]:
        """Compute high-level graph quality metrics: coherence, coverage, redundancy and semantic density."""
        if not nodes:
            return {
                'coherence_score': 0.0,
                'semantic_coverage': 0.0,
                'redundancy': 0.0,
                'semantic_density': 0.0,
                'quality_score': 0.0,
            }

        non_root_nodes = [n for n in nodes if n.get('level', 0) > 0]
        normalized_labels = [self.nlp_engine._normalize_term(n.get('label', '')) for n in non_root_nodes]
        normalized_labels = [l for l in normalized_labels if l]

        unique_labels = set(normalized_labels)
        exact_redundancy = 1.0 - (len(unique_labels) / max(1, len(normalized_labels)))

        # Additional overlap redundancy from high lexical overlap between node labels
        overlap_pairs = 0
        total_pairs = 0
        for i in range(len(normalized_labels)):
            a = set(normalized_labels[i].split())
            if not a:
                continue
            for j in range(i + 1, len(normalized_labels)):
                b = set(normalized_labels[j].split())
                if not b:
                    continue
                total_pairs += 1
                jacc = len(a & b) / max(1, len(a | b))
                if jacc >= 0.8:
                    overlap_pairs += 1

        overlap_redundancy = (overlap_pairs / max(1, total_pairs)) if total_pairs else 0.0
        redundancy = min(1.0, (0.7 * exact_redundancy) + (0.3 * overlap_redundancy))

        # Semantic coverage: fraction of sentences represented by at least one graph concept label
        sentence_spans = self.nlp_engine._split_sentences_with_spans(text)
        represented_sentences = 0
        for sent, _ in sentence_spans:
            sent_norm = self.nlp_engine._normalize_unicode(sent)
            hit = False
            for label in unique_labels:
                if label and label in sent_norm:
                    hit = True
                    break
            if hit:
                represented_sentences += 1

        semantic_coverage = represented_sentences / max(1, sentence_count)
        semantic_density = len(non_root_nodes) / max(1, sentence_count)

        coherence_score = float(coherence_metrics.get('coherence_score', 0.0))

        # Quality score prefers coherent, high-coverage, low-redundancy graphs
        quality_score = (
            (0.4 * coherence_score)
            + (0.35 * semantic_coverage)
            + (0.15 * min(1.0, semantic_density / 3.0))
            + (0.10 * max(0.0, 1.0 - redundancy))
        )

        return {
            'coherence_score': round(float(coherence_score), 4),
            'semantic_coverage': round(float(semantic_coverage), 4),
            'redundancy': round(float(redundancy), 4),
            'semantic_density': round(float(semantic_density), 4),
            'quality_score': round(float(quality_score), 4),
        }

    def _compute_confidence_distribution(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Track confidence distribution across nodes and edges."""

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {
                    'min': 0.0,
                    'max': 0.0,
                    'mean': 0.0,
                    'median': 0.0,
                    'p10': 0.0,
                    'p90': 0.0,
                    'low_ratio': 0.0,
                    'mid_ratio': 0.0,
                    'high_ratio': 0.0,
                }
            vals = sorted(max(0.0, min(1.0, float(v))) for v in values)
            n = len(vals)
            mean = sum(vals) / n
            median = vals[n // 2] if n % 2 == 1 else (vals[(n // 2) - 1] + vals[n // 2]) / 2.0
            p10 = vals[max(0, int(0.10 * (n - 1)))]
            p90 = vals[min(n - 1, int(0.90 * (n - 1)))]
            low = sum(1 for v in vals if v < 0.45) / n
            mid = sum(1 for v in vals if 0.45 <= v < 0.7) / n
            high = sum(1 for v in vals if v >= 0.7) / n
            return {
                'min': round(float(vals[0]), 4),
                'max': round(float(vals[-1]), 4),
                'mean': round(float(mean), 4),
                'median': round(float(median), 4),
                'p10': round(float(p10), 4),
                'p90': round(float(p90), 4),
                'low_ratio': round(float(low), 4),
                'mid_ratio': round(float(mid), 4),
                'high_ratio': round(float(high), 4),
            }

        node_conf = [float(n.get('importance', 0.0)) for n in nodes if n.get('level', 0) > 0]
        edge_conf = []
        for e in edges:
            if 'confidence' in e:
                edge_conf.append(float(e.get('confidence', 0.0)))
            else:
                edge_conf.append(max(0.0, min(1.0, float(e.get('weight', 1)) / 3.0)))

        return {
            'nodes': _stats(node_conf),
            'edges': _stats(edge_conf),
        }

    def _build_low_confidence_explanations(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate explanations for low-confidence node/edge extractions."""
        node_issues: List[Dict[str, Any]] = []
        edge_issues: List[Dict[str, Any]] = []

        for node in nodes:
            if node.get('level', 0) == 0:
                continue
            conf = float(node.get('importance', 0.0))
            if conf >= 0.45:
                continue
            source_type = node.get('source_type', 'unknown')
            reason = 'Low salience in essay context'
            if source_type == 'enumeration_item':
                reason = 'Derived from list structure with limited semantic support'
            elif source_type == 'phrase_extraction':
                reason = 'Phrase prominence is below strong concept threshold'
            elif source_type == 'subtopic':
                reason = 'Subtopic has weak entity importance signal'
            node_issues.append({
                'id': node.get('id'),
                'label': node.get('label'),
                'confidence': round(conf, 4),
                'reason': reason,
                'suggestion': 'Review surrounding sentence or merge with a stronger related concept.'
            })

        for edge in edges:
            conf = float(edge.get('confidence', max(0.0, min(1.0, float(edge.get('weight', 1)) / 3.0))))
            if conf >= 0.5:
                continue
            edge_type = edge.get('type', 'related')
            reason = 'Weak relation evidence between connected concepts'
            if edge_type in {'semantic', 'related-to', 'proximity', 'conjunction', 'close'}:
                reason = 'Similarity/proximity signal is weak and may be contextual noise'
            elif edge_type in {'subitem_of', 'includes'}:
                reason = 'List-derived link has limited context support'
            edge_issues.append({
                'id': edge.get('id'),
                'type': edge_type,
                'source': edge.get('source'),
                'target': edge.get('target'),
                'confidence': round(conf, 4),
                'reason': reason,
                'suggestion': 'Keep only if this relation is pedagogically meaningful.'
            })

        # Keep metadata compact while still informative
        return {
            'nodes': node_issues[:20],
            'edges': edge_issues[:30],
            'summary': {
                'low_conf_nodes': len(node_issues),
                'low_conf_edges': len(edge_issues),
            }
        }

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
    
    def _clean_label_preserving(self, text: str) -> str:
        """Clean label while preserving Sinhala text integrity and meaning."""
        if not text or not text.strip():
            return text
        
        # Normalize unicode but preserve Sinhala characters
        text = unicodedata.normalize('NFC', text)
        text = text.strip()
        
        # Only remove obvious noise, keep meaningful Sinhala words
        words = text.split()
        
        # Filter strategy: keep all non-empty words unless they're pure stop words
        cleaned_words = []
        for word in words:
            word = word.strip()
            # Skip empty or very short noise
            if len(word) <= 1:
                continue
            # Keep if not in stop word list or if it's substantial (>3 chars)
            if word not in self.nlp_engine.stop_words or len(word) > 3:
                cleaned_words.append(word)
        
        # If we filtered everything, keep original
        if not cleaned_words:
            return text
        
        # Limit to 4 words max for readability
        result = ' '.join(cleaned_words[:4])
        
        # Ensure we didn't accidentally create an empty label
        return result if result.strip() else text
    
    def _empty_graph(self) -> Dict[str, Any]:
        """Return an empty graph structure."""
        return {
            'nodes': [],
            'edges': [],
            'metadata': {
                'total_nodes': 0,
                'total_edges': 0,
                'quality': {
                    'coherence_score': 0.0,
                    'semantic_coverage': 0.0,
                    'redundancy': 0.0,
                    'semantic_density': 0.0,
                    'quality_score': 0.0,
                },
                'confidence_distribution': {
                    'nodes': {},
                    'edges': {},
                },
                'semantic_density': 0.0,
                'semantic_coverage': 0.0,
                'redundancy': 0.0,
                'low_confidence_explanations': {
                    'nodes': [],
                    'edges': [],
                    'summary': {'low_conf_nodes': 0, 'low_conf_edges': 0},
                },
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

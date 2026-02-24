"""
Advanced NLP Engine for Sinhala Text Processing
Uses lightweight models for intelligent node and relationship extraction.
"""

import re
import hashlib
import math
import unicodedata
from typing import List, Dict, Tuple, Set, Any
import numpy as np
from collections import defaultdict, Counter
import logging
from config import Config
from hybrid_node_extractor import HybridNodeExtractor
from relation_classifier import RelationClassifier

logger = logging.getLogger(__name__)


class SinhalaNLPEngine:
    """Lightweight NLP engine for Sinhala text processing."""

    def __init__(self):
        """Initialize the NLP engine with lightweight models."""
        self.embeddings_model = None
        self.sinling_available = False
        self.sinling_tokenizer = None
        self.embeddings_model_name = getattr(
            Config, 'EMBEDDINGS_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # Initialize sentence transformer for semantic analysis
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight multilingual model that supports Sinhala
            self.embeddings_model = SentenceTransformer(self.embeddings_model_name)
            logger.info("Loaded multilingual embeddings model: %s", self.embeddings_model_name)
        except Exception as e:
            logger.warning("Could not load embeddings model %s: %s", self.embeddings_model_name, e)
        
        # Try to import sinling for Sinhala NLP
        try:
            import sinling
            self.sinling_available = True
            logger.info("Sinling library available for Sinhala processing")
            try:
                from sinling import SinhalaTokenizer
                self.sinling_tokenizer = SinhalaTokenizer()
            except Exception as e:
                logger.warning("Could not initialize SinhalaTokenizer: %s", e)
        except ImportError:
            logger.warning("Sinling not available, using fallback methods")
        
        # Sinhala linguistic patterns
        self.postpositions = ['සඳහා', 'විසින්', 'ගැන', 'පිළිබඳව', 'අතර', 'තුළ', 'හි', 'සමග', 'සමඟ', 'සිට', 'දක්වා', 'වන්', 'වන', 'මගින්']
        self.connectors = ['සහ', 'හා', 'මෙන්ම', 'ද', 'ඒ', 'වගේම', 'පමණක්', 'ඉතා', 'වඩා', 'මෙන්']
        self.verbs_indicators = ['කරන', 'වන', 'ඇති', 'යන', 'බව', 'නිසා', 'ලබන', 'දෙන', 'වේ', 'වී', 'ගන්නා', 'දෙන']
        
        # Action verbs and weak words to filter (not pure concepts)
        self.action_verbs = ['නිපදවන', 'ඉස්සරවන', 'ඇතිවේ', 'බිඳිනවා', 'හසුරුවන', 'ගිය', 'එන', 'යන', 'කරයි', 'කරණ', 'සිදු', 'ගිනේ']
        self.weak_words = ['වුණු', 'වුණා', 'ඇතුළු', 'වන', 'වේ', 'වැයි', 'වෙයි', 'ඉතිරිය', 'ගුණඉංගිතවතුන්', 'ජීවිතයේ', 'කෝශවල', 'ප්‍රධාන', 'කොටස්ය']
        
        self.question_words = ['කවර', 'කොහොමද', 'කුමක්', 'ඇයි', 'මොකද', 'කවදා', 'කොතැනද']

        # Relationship markers for heuristic edge typing
        self.definition_markers = ['යනු', 'හෙවත්', 'කියන', 'අර්ථය', 'අර්ථයෙන්', 'හැඳින්වෙයි']
        self.cause_markers = ['නිසා', 'හේතුවෙන්', 'ප්‍රතිඵලයෙන්', 'ප්‍රතිඵලයක් ලෙස', 'ඒ හේතුවෙන්']
        self.part_markers = ['අංශය', 'කොටස', 'අයත්', 'අංග', 'අභ්‍යන්තර', 'භාගය']
        self.example_markers = ['උදාහරණ', 'උදාහරණයක්', 'උදා:', 'දෙසට']

        # Tokenization helpers
        self.word_pattern = re.compile(r'[A-Za-z0-9\u0D80-\u0DFF]+')
        self.list_split_pattern = re.compile(r'\s*,\s*|\s+සහ\s+|\s+හා\s+|\s+මෙන්ම\s+')
        self.sequence_markers = ['අදියර', 'පියවර', 'අවස්ථා']
        self.importance_markers = ['වැදගත්', 'අත්‍යවශ්‍ය', 'අතිශය වැදගත්']
        
        # Comprehensive Sinhala stop words (helping words that should not appear in nodes)
        self.stop_words = set([
            # Pronouns
            'මම', 'ඔබ', 'ඔහු', 'ඇය', 'අපි', 'ඔවුන්', 'මා', 'අප', 'ඔබට', 'මට',
            # Articles and determiners
            'එක්', 'එක', 'මේ', 'මෙම', 'ඒ', 'ඔය', 'ඕ', 'කිසිවක්', 'සියලු', 'සියළු',
            # Common helping verbs
            'වන', 'වන්නේ', 'වේ', 'වී', 'වෙයි', 'වුණා', 'වෙන්නේ', 'ඇති', 'ඇත', 'ඇත්තේ',
            # Postpositions
            'සඳහා', 'විසින්', 'ගැන', 'පිළිබඳව', 'අතර', 'තුළ', 'හි', 'සමග', 'සමඟ', 'සිට', 'දක්වා', 'මගින්',
            # Conjunctions
            'සහ', 'හා', 'මෙන්ම', 'ද', 'වගේම', 'පමණක්', 'නමුත්', 'කෙසේ', 'හෝ',
            # Auxiliary words
            'ඉතා', 'වඩා', 'මෙන්', 'තරම්', 'පමණ', 'ලෙස', 'ලෙසින්', 'යනාදී', 'ආදිය',
            # Common particles
            'ද', 'ය', 'ක්', 'යි', 'නේ', 'නේද', 'ඩ', 'නම්',
            # Question particles
            'කවර', 'කොහොමද', 'කුමක්', 'ඇයි', 'මොකද', 'කවදා', 'කොතැනද', 'කොහෙද',
            # Modal particles  
            'බව', 'නිසා', 'නම්', 'යැයි', 'කියා',
            # Short common words
            'ට', 'ටත්', 'යන', 'දී', 'ගේ', 'වල', 'ය', 'යි'
        ])

        # Hybrid node extractor (rule patterns + embeddings + re-ranker)
        self.hybrid_extractor = HybridNodeExtractor(self)

        # Two-stage relation classifier (candidate generation + feature classification)
        self.relation_classifier = RelationClassifier(self)

        # ── Per-relation-type confidence thresholds ───────────────────────────
        # Rationale:
        #   is-a        – requires an explicit definition / example marker; set high
        #                 so only linguistically-grounded definitional edges survive.
        #   part-of     – part/component markers are fairly reliable but can appear
        #                 in loose contexts; moderate threshold.
        #   cause-effect– causal claims need strong lexical evidence; set high.
        #   related-to  – catch-all (proximity, conjunction, semantic co-occurrence);
        #                 set low to preserve weak but plausible links.
        self.rel_thresholds: Dict[str, float] = {
            'is-a':         0.58,
            'part-of':      0.52,
            'cause-effect': 0.55,
            'related-to':   0.42,
        }

        # Map internal relation labels → canonical four-way taxonomy
        self.rel_type_canonical: Dict[str, str] = {
            'definition':  'is-a',
            'example':     'is-a',
            'part_of':     'part-of',
            'cause':       'cause-effect',
            'action':      'cause-effect',
            'semantic':    'related-to',
            'conjunction': 'related-to',
            'close':       'related-to',
            'related':     'related-to',
            'proximity':   'related-to',
        }

    # ------------------------------------------------------------------
    # Public extraction methods
    # ------------------------------------------------------------------

    def extract_nodes_hybrid(self, text: str, max_nodes: int = 40) -> List[Dict[str, Any]]:
        """
        Hybrid concept node extraction combining:
          1. Rule-based POS / chunk patterns (high recall)
          2. Embedding similarity to concept anchors (high precision)
          3. Confidence re-ranker (balanced ranking)

        Returns
        -------
        List[Dict]
            Sorted by ``confidence`` descending. Each dict contains:
            ``text``, ``type``, ``importance``, ``confidence``,
            ``rule_score``, ``embed_score``, ``frequency``,
            ``context``, ``offset``, ``normalized``
        """
        from sinhala_normalization import normalize_sinhala_text
        text = normalize_sinhala_text(self._normalize_unicode(text))
        return self.hybrid_extractor.extract(text, max_nodes=max_nodes)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities and important concepts from text.

        Delegates to the HybridNodeExtractor pipeline (rule patterns +
        embedding similarity + confidence re-ranker) for improved recall
        and precision over the previous single-signal heuristic approach.

        Returns
        -------
        List[Dict]
            Entity dicts sorted by ``importance`` (= ``confidence``) descending.
        """
        from sinhala_normalization import normalize_sinhala_text
        text = normalize_sinhala_text(self._normalize_unicode(text))
        entities = self.hybrid_extractor.extract(text)
        # Classify entity type for any node that still has the default type
        for e in entities:
            if e.get('type') in ('concept', None):
                e['type'] = self._classify_entity_type(e['text'], e.get('context', ''))
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using a two-stage pipeline:

        Stage 1 – Candidate generation
            Shortlists node pairs via same/adjacent sentence co-occurrence,
            embedding similarity, and shared n-gram root — avoiding the O(n²)
            complexity of exhaustive pair enumeration.

        Stage 2 – Relation classification
            For each shortlisted pair, builds a 14-dim feature vector
            (lexical markers × context scope, embedding similarity, between-span
            type similarity, structural distance) and scores it against a
            calibrated weight matrix.  A softmax gives a probability distribution
            over the four canonical types; the argmax is the predicted relation
            and its probability is the confidence.

        Per-type thresholds (``self.rel_thresholds``) are applied after
        classification to suppress weak predictions.

        Returns
        -------
        List[Dict]
            Relationship dicts with keys: ``source``, ``target``, ``type``,
            ``confidence``, ``context``, ``feature_scores``.
        """
        from sinhala_normalization import normalize_sinhala_text
        text = normalize_sinhala_text(self._normalize_unicode(text))
        try:
            return self.relation_classifier.classify(text, entities)
        except Exception as exc:
            logger.warning(
                "RelationClassifier failed (%s); falling back to heuristic.", exc
            )
            return self._extract_relationships_heuristic(text, entities)

    def _extract_relationships_heuristic(
        self, text: str, entities: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Legacy heuristic relationship extraction (fallback only).

        Used when the supervised relation classifier raises an unexpected error.
        """
        relationships = []
        sentences = self._split_sentences_with_spans(text)
        rel_keys: set = set()

        for sentence, sent_start in sentences:
            sent_end = sent_start + len(sentence)
            sentence_entities = [
                e for e in entities
                if e.get('offset', -1) >= sent_start and e.get('offset', -1) < sent_end
            ]
            if len(sentence_entities) < 2:
                continue
            for i in range(len(sentence_entities)):
                for j in range(i + 1, len(sentence_entities)):
                    e1, e2 = sentence_entities[i], sentence_entities[j]
                    raw_type, confidence = self._analyze_relationship(
                        e1['text'], e2['text'], sentence,
                        e1.get('offset'), e2.get('offset')
                    )
                    rel_type = self.rel_type_canonical.get(raw_type, 'related-to')
                    threshold = self.rel_thresholds.get(rel_type, 0.45)
                    if confidence <= threshold:
                        continue
                    key = tuple(sorted([e1['text'], e2['text']]) + [rel_type])
                    if key not in rel_keys:
                        rel_keys.add(key)
                        relationships.append({
                            'source': e1['text'], 'target': e2['text'],
                            'type': rel_type, 'confidence': confidence,
                            'context': sentence,
                        })
        return relationships
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Returns:
            Similarity score between 0 and 1
        """
        if not self.embeddings_model:
            # Fallback to lexical and character n-gram similarity
            return max(
                self._word_overlap_similarity(text1, text2),
                self._char_ngram_similarity(text1, text2)
            )
        
        try:
            embeddings = self.embeddings_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return max(
                self._word_overlap_similarity(text1, text2),
                self._char_ngram_similarity(text1, text2)
            )
    
    def cluster_concepts(self, entities: List[Dict], threshold: float = 0.6) -> List[List[Dict]]:
        """
        Cluster similar entities together based on semantic similarity.
        
        Returns:
            List of entity clusters
        """
        if not entities:
            return []
        
        if not self.embeddings_model:
            # Simple clustering based on word patterns
            return self._simple_cluster(entities)
        
        try:
            # Extract texts for embedding
            texts = [e['text'] for e in entities]
            embeddings = self.embeddings_model.encode(texts)

            threshold = self._adaptive_cluster_threshold(embeddings, threshold)
            
            # Simple agglomerative clustering
            clusters = []
            used = set()
            
            for i, entity in enumerate(entities):
                if i in used:
                    continue
                
                cluster = [entity]
                used.add(i)
                
                for j in range(i + 1, len(entities)):
                    if j in used:
                        continue
                    
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    
                    if similarity > threshold:
                        cluster.append(entities[j])
                        used.add(j)
                
                clusters.append(cluster)
            
            return clusters
        except Exception as e:
            logger.warning(f"Error clustering: {e}")
            return [[e] for e in entities]
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
        from sinhala_normalization import normalize_sinhala_text
        text = normalize_sinhala_text(text)
        """Sinhala-aware phrase chunking: extract noun phrases and named entities."""
        if not text:
            return []
        sentences = self._split_sentences(text)
        phrases = []
        for sentence in sentences:
            # Use sinling noun phrase chunker if available
            try:
                from sinling import SinhalaPhraseChunker
                chunker = SinhalaPhraseChunker()
                chunks = chunker.chunk(sentence)
                for chunk in chunks:
                    if chunk['type'] == 'NP':
                        phrase = chunk['text']
                        if len(phrase) >= 8 and len(phrase) <= 80 and not self._is_stop_phrase(phrase):
                            score = self._calculate_phrase_importance(phrase, text)
                            if score > 0.3:
                                phrases.append((phrase, score))
                continue
            except Exception:
                pass
            # Fallback: use sliding window for multi-word noun-like phrases
            words = self._tokenize(sentence)
            for i in range(len(words)):
                if words[i] in self.stop_words or len(words[i]) <= 2:
                    continue
                for j in range(i + 2, min(i + 4, len(words) + 1)):
                    phrase = ' '.join(words[i:j])
                    if len(phrase) >= 8 and len(phrase) <= 80 and not self._is_stop_phrase(phrase):
                        score = self._calculate_phrase_importance(phrase, text)
                        if score > 0.3:
                            phrases.append((phrase, score))
        # Deduplicate and sort
        phrase_dict = {}
        for phrase, score in phrases:
            if phrase not in phrase_dict or phrase_dict[phrase] < score:
                phrase_dict[phrase] = score
        phrase_list = sorted(phrase_dict.items(), key=lambda x: x[1], reverse=True)
        filtered_phrases = []
        for phrase, score in phrase_list:
            is_subphrase = False
            for existing, _ in filtered_phrases:
                if phrase in existing and len(existing) > len(phrase):
                    is_subphrase = True
                    break
            if not is_subphrase:
                filtered_phrases.append((phrase, score))
        return filtered_phrases[:max_phrases]

    def extract_enumerations(self, text: str) -> List[Dict[str, Any]]:
        """Extract enumerations like phases, examples, and required items."""
        enumerations: List[Dict[str, Any]] = []
        sentences = self._split_sentences_with_spans(text)

        for sentence, _ in sentences:
            if not sentence or len(sentence) < 10:
                continue

            # Phase/sequence enumeration
            if any(marker in sentence for marker in self.sequence_markers):
                list_text = self._extract_list_after_marker(sentence, 'ලෙස') or self._extract_list_after_marker(sentence, 'වශයෙන්')
                if not list_text:
                    list_text = sentence
                items = self._extract_list_items(list_text)
                if len(items) >= 2:
                    head = self._infer_head_phrase(sentence, self.sequence_markers)
                    enumerations.append({
                        'head': head or 'අදියර',
                        'items': items,
                        'relation': 'sequence'
                    })

            # Example enumeration with "වැනි" or explicit examples
            if 'වැනි' in sentence or any(marker in sentence for marker in self.example_markers):
                before, after = sentence.split('වැනි', 1) if 'වැනි' in sentence else (sentence, '')
                items = self._extract_list_items(before)
                head = self._first_token(after) or self._infer_head_phrase(sentence, ['භාෂා', 'ක්‍රමවේද', 'මූලධර්ම'])
                if len(items) >= 2:
                    enumerations.append({
                        'head': head,
                        'items': items,
                        'relation': 'example'
                    })

            # Importance/requirements enumeration
            if any(marker in sentence for marker in self.importance_markers):
                items = self._extract_list_items(sentence)
                if len(items) >= 2:
                    enumerations.append({
                        'head': None,
                        'items': items,
                        'relation': 'requires'
                    })

        return enumerations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (legacy helper)."""
        return [s for s, _ in self._split_sentences_with_spans(text)]

    def _split_sentences_with_spans(self, text: str) -> List[Tuple[str, int]]:
        """Split text into sentences and return (sentence, start_index)."""
        delimiters = ['.', '|', '?', '!', '।', ':', '\n']
        pattern = '|'.join(map(re.escape, delimiters))
        sentences = []
        start = 0
        for match in re.finditer(pattern, text + ' '):  # add trailing space to catch last segment
            end = match.start()
            segment = text[start:end].strip()
            if segment and len(segment) > 5:
                sentences.append((segment, start))
            start = match.end()
        # Trailing text after last delimiter
        if start < len(text):
            tail = text[start:].strip()
            if tail and len(tail) > 5:
                sentences.append((tail, start))
        return sentences
    
    def _calculate_word_importance(self, word: str, sentence: str, position: int, total_words: int) -> float:
        """Calculate importance score for a word. Prioritizes nouns over verbs."""
        # Filter out stop words completely
        if word in self.stop_words or word in self.postpositions or word in self.connectors:
            return 0.0  # Return 0 to filter out helping words
        
        # Filter out action verbs and weak words (not pure concepts)
        if word in self.action_verbs or word in self.weak_words:
            return 0.0
        
        # Check if word contains verb patterns (lower importance)
        if any(verb in word for verb in self.verbs_indicators):
            # Words like 'නිපදවන' have verb endings, give lower importance
            if word.endswith('න') or word.endswith('නවා') or word.endswith('ටන'):
                return 0.2  # Very low importance for verb forms
        
        # Filter out very short words (likely particles)
        if len(word) <= 2:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length factor (longer words often more important in Sinhala)
        if len(word) >= 5:
            score += 0.2
        if len(word) >= 8:
            score += 0.1
        
        # Position factor (beginning and end of sentences often important)
        if position < 2 or position >= total_words - 2:
            score += 0.15
        
        return min(1.0, max(0.0, score))
    
    def _is_stop_phrase(self, phrase: str) -> bool:
        """Check if phrase consists mainly of stop words."""
        words = phrase.split()
        if not words:
            return True
        
        # Count how many words are stop words
        stop_count = sum(1 for word in words if word in self.stop_words or len(word) <= 2)
        
        # If more than 60% are stop words, consider the phrase as a stop phrase
        return stop_count / len(words) > 0.6
    
    def _classify_entity_type(self, word: str, context: str) -> str:
        """Classify the type of entity."""
        # Check if it's a weak word or action verb
        if word in self.action_verbs or word in self.weak_words:
            return 'verb'
        
        # Simple classification based on patterns
        if any(q in context for q in self.question_words):
            return 'query'
        elif any(verb in word for verb in self.verbs_indicators):
            # Check word ending for verb forms
            if word.endswith('න') or word.endswith('ටන') or word.endswith('ෙයි'):
                return 'verb'
            return 'action'
        elif word in self.postpositions:
            return 'relation'
        else:
            return 'concept'
    
    def _analyze_relationship(self, entity1: str, entity2: str, sentence: str,
                              offset1: int = None, offset2: int = None) -> Tuple[str, float]:
        """Analyze the relationship between two entities in a sentence."""
        # Find the text between entities
        idx1 = sentence.find(entity1)
        idx2 = sentence.find(entity2)
        
        if idx1 == -1 or idx2 == -1:
            return 'related', 0.3
        
        start = min(idx1, idx2) + len(entity1 if idx1 < idx2 else entity2)
        end = max(idx1, idx2)
        between = sentence[start:end].strip()
        
        confidence = 0.5
        rel_type = 'related'
        
        # Proximity boost: closer entities imply stronger link
        if offset1 is not None and offset2 is not None:
            distance = abs(offset1 - offset2)
            proximity_score = max(0.0, 1.0 - min(distance, 200) / 200.0)  # clamp at 200 chars
            confidence += 0.2 * proximity_score
        
        # Check for specific relationship indicators (ordered by specificity)
        if any(marker in between for marker in self.definition_markers):
            rel_type = 'definition'
            confidence += 0.35
        elif any(marker in between for marker in self.cause_markers):
            rel_type = 'cause'
            confidence += 0.3
        elif any(marker in between for marker in self.part_markers):
            rel_type = 'part_of'
            confidence += 0.25
        elif any(marker in between for marker in self.example_markers):
            rel_type = 'example'
            confidence += 0.2
        elif any(post in between for post in self.postpositions):
            rel_type = 'semantic'
            confidence += 0.2
        elif any(conn in between for conn in self.connectors):
            rel_type = 'conjunction'
            confidence += 0.3
        elif any(verb in between for verb in self.verbs_indicators):
            rel_type = 'action'
            confidence += 0.25
        elif len(between.split()) <= 3:
            rel_type = 'close'
            confidence += 0.1
        
        return rel_type, min(confidence, 1.0)
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities, keeping highest importance.
        
        Also filters out pure verb forms and weak words.
        """
        entity_dict = {}
        
        for entity in entities:
            text = entity['text']
            key = entity.get('normalized') or self._normalize_term(text)
            if not key:
                key = text
            
            # Skip if it's a verb form or weak word
            if text in self.action_verbs or text in self.weak_words:
                continue
            
            # Skip if it's mostly verb
            if text.endswith('න') or text.endswith('ටන') or text.endswith('ෙයි'):
                # But allow if it's a well-known concept
                if text not in ['නිපදවන', 'ඉස්සරවන', 'ඇතිවේ', 'බිඳිනවා']:
                    if text not in entity_dict or entity_dict[text]['importance'] < entity['importance']:
                        entity_dict[text] = entity
                continue
            
            if key not in entity_dict or entity_dict[key]['importance'] < entity['importance']:
                entity_dict[key] = entity
                entity_dict[key]['aliases'] = list({text} | set(entity.get('aliases', [])))
            else:
                aliases = set(entity_dict[key].get('aliases', []))
                aliases.add(text)
                entity_dict[key]['aliases'] = list(aliases)
        
        return list(entity_dict.values())
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    def _char_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Character n-gram similarity for short Sinhala terms."""
        t1 = self._normalize_term(text1)
        t2 = self._normalize_term(text2)

        if not t1 or not t2:
            return 0.0

        grams1 = {t1[i:i + n] for i in range(max(1, len(t1) - n + 1))}
        grams2 = {t2[i:i + n] for i in range(max(1, len(t2) - n + 1))}
        if not grams1 or not grams2:
            return 0.0

        intersection = len(grams1 & grams2)
        union = len(grams1 | grams2)
        return intersection / union if union else 0.0
    
    def clean_label(self, text: str) -> str:
        """Remove stop words from text to show only concepts."""
        if not text or not text.strip():
            return text
        text = self._normalize_unicode(text)
        words = text.split()
        # Filter out stop words and very short words
        cleaned_words = [
            word for word in words 
            if word not in self.stop_words 
            and len(word) > 2
            and word not in self.postpositions
            and word not in self.connectors
        ]
        
        # If all words were filtered out, keep longest words as fallback
        if not cleaned_words:
            # Sort by length and keep the longest ones
            words_by_length = sorted(words, key=len, reverse=True)
            cleaned_words = words_by_length[:min(3, len(words_by_length))]
        
        return ' '.join(cleaned_words)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text with Sinhala-aware pattern (sinling if available, else regex)."""
        if not text:
            return []
        if self.sinling_tokenizer:
            try:
                tokens = self.sinling_tokenizer.tokenize(text)
                return [t for t in tokens if t and t.strip()]
            except Exception:
                pass
        # Remove zero-width characters then extract Sinhala and Latin tokens
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        return self.word_pattern.findall(text)

    def _adaptive_cluster_threshold(self, embeddings: np.ndarray, base_threshold: float) -> float:
        """Adjust clustering threshold based on similarity distribution."""
        if embeddings is None or len(embeddings) < 3:
            return base_threshold

        sample_pairs = []
        max_pairs = 120
        total = len(embeddings)
        for i in range(total):
            for j in range(i + 1, min(total, i + 10)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                sample_pairs.append(sim)
                if len(sample_pairs) >= max_pairs:
                    break
            if len(sample_pairs) >= max_pairs:
                break

        if not sample_pairs:
            return base_threshold

        median_sim = float(np.median(sample_pairs))
        return max(base_threshold, min(0.85, median_sim + 0.08))

    def _extract_list_after_marker(self, sentence: str, marker: str) -> str:
        """Extract list-like substring after a marker like 'ලෙස' or 'වශයෙන්'."""
        if marker in sentence:
            parts = sentence.split(marker, 1)
            if len(parts) == 2:
                return parts[1]
        return ''

    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from a sentence-like fragment."""
        if not text:
            return []
        cleaned = re.sub(r'[()\[\]{}"\'\u201c\u201d]', ' ', text)
        candidates = [c.strip() for c in self.list_split_pattern.split(cleaned) if c.strip()]
        items = []
        for cand in candidates:
            label = self.clean_label(cand)
            if label and len(label) >= 2 and label not in items:
                items.append(label)
        return items

    def _infer_head_phrase(self, sentence: str, keywords: List[str]) -> str:
        """Infer a head concept phrase around a keyword."""
        for keyword in keywords:
            if keyword in sentence:
                idx = sentence.find(keyword)
                left = sentence[max(0, idx - 30):idx].strip()
                tokens = [t for t in self._tokenize(left) if t not in self.stop_words]
                if tokens:
                    return ' '.join(tokens[-2:] + [keyword]) if keyword not in tokens else ' '.join(tokens[-2:])
                return keyword
        return ''

    def _first_token(self, text: str) -> str:
        """Return the first token in text."""
        tokens = self._tokenize(text)
        return tokens[0] if tokens else ''

    def _sentence_document_frequency(self, sentences: List[Tuple[str, int]]) -> Tuple[Dict[str, int], int]:
        """Calculate sentence-level document frequency for normalized tokens."""
        df_map = Counter()
        for sentence, _ in sentences:
            tokens = {self._normalize_term(t) for t in self._tokenize(sentence)}
            tokens = {t for t in tokens if t}
            for token in tokens:
                df_map[token] += 1
        return dict(df_map), len(sentences)

    def _normalize_term(self, text: str) -> str:
        """Normalize term for comparison and deduping."""
        if not text:
            return ''
        text = self._normalize_unicode(text)
        cleaned = re.sub(r'[^A-Za-z0-9\u0D80-\u0DFF ]+', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode to NFC to preserve Sinhala combining marks."""
        try:
            return unicodedata.normalize('NFC', text)
        except Exception:
            return text
    
    def _simple_cluster(self, entities: List[Dict]) -> List[List[Dict]]:
        """Simple clustering based on text patterns."""
        clusters = []
        used = set()
        
        for i, entity in enumerate(entities):
            if i in used:
                continue
            
            cluster = [entity]
            used.add(i)
            
            for j in range(i + 1, len(entities)):
                if j in used:
                    continue
                
                # Check for similarity
                similarity = max(
                    self._word_overlap_similarity(entity['text'], entities[j]['text']),
                    self._char_ngram_similarity(entity['text'], entities[j]['text'])
                )
                
                if similarity > 0.3:
                    cluster.append(entities[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_phrase_importance(self, phrase: str, full_text: str) -> float:
        """Calculate importance score for a phrase."""
        score = 0.5
        
        # Frequency in text
        count = full_text.count(phrase)
        if count > 1:
            score += min(0.3, count * 0.1)
        
        # Length factor
        word_count = len(phrase.split())
        if 3 <= word_count <= 5:
            score += 0.2
        
        # Check if contains important words
        important_words = [w for w in phrase.split() if len(w) >= 5]
        score += min(0.2, len(important_words) * 0.05)
        
        return min(1.0, score)

    def _find_offset(self, sentence: str, word: str, sentence_start: int, used_spans: List[Tuple[int, int]]) -> int:
        """Find a stable character offset for the word within the full text."""
        pattern = re.escape(word)
        for match in re.finditer(pattern, sentence):
            start = sentence_start + match.start()
            end = sentence_start + match.end()
            if not any(us <= start < ue or us < end <= ue for us, ue in used_spans):
                return start
        # Fallback to sentence start if not found uniquely
        return sentence_start

"""
Advanced NLP Engine for Sinhala Text Processing
Uses lightweight models for intelligent node and relationship extraction.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Set, Any
import numpy as np
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class SinhalaNLPEngine:
    """Lightweight NLP engine for Sinhala text processing."""
    
    def __init__(self):
        """Initialize the NLP engine with lightweight models."""
        self.embeddings_model = None
        self.sinling_available = False
        
        # Initialize sentence transformer for semantic analysis
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight multilingual model that supports Sinhala
            self.embeddings_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Loaded multilingual embeddings model")
        except Exception as e:
            logger.warning(f"Could not load embeddings model: {e}")
        
        # Try to import sinling for Sinhala NLP
        try:
            import sinling
            self.sinling_available = True
            logger.info("Sinling library available for Sinhala processing")
        except ImportError:
            logger.warning("Sinling not available, using fallback methods")
        
        # Sinhala linguistic patterns
        self.postpositions = ['සඳහා', 'විසින්', 'ගැන', 'පිළිබඳව', 'අතර', 'තුළ', 'හි', 'සමග', 'සමඟ']
        self.connectors = ['සහ', 'හා', 'මෙන්ම', 'ද', 'ඒ', 'වගේම', 'පමණක්']
        self.verbs_indicators = ['කරන', 'වන', 'ඇති', 'යන', 'බව', 'නිසා', 'ලබන', 'දෙන']
        self.question_words = ['කවර', 'කොහොමද', 'කුමක්', 'ඇයි', 'මොකද', 'කවදා', 'කොතැනද']
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities and important concepts from text.
        
        Returns:
            List of entity dictionaries with type, text, and importance score
        """
        entities = []
        
        # Split text into sentences with character offsets for positional tracking
        sentences = self._split_sentences_with_spans(text)
        used_spans: List[Tuple[int, int]] = []
        
        for sentence, sent_start in sentences:
            # Extract proper nouns (capitalized words in Sinhala context)
            words = sentence.split()
            
            for i, word in enumerate(words):
                # Check if word is a potential entity
                if len(word) >= 3:
                    # Calculate importance based on position and context
                    importance = self._calculate_word_importance(word, sentence, i, len(words))
                    
                    if importance > 0.3:
                        # Track a stable character offset for this occurrence
                        offset = self._find_offset(sentence, word, sent_start, used_spans)
                        used_spans.append((offset, offset + len(word)))
                        entities.append({
                            'text': word,
                            'type': self._classify_entity_type(word, sentence),
                            'importance': importance,
                            'context': sentence,
                            'offset': offset
                        })
        
        # Deduplicate and sort by importance
        entities = self._deduplicate_entities(entities)
        return sorted(entities, key=lambda x: x['importance'], reverse=True)
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Returns:
            List of relationship dictionaries with source, target, type, and confidence
        """
        relationships = []
        sentences = self._split_sentences_with_spans(text)
        entity_texts = {e['text'] for e in entities}
        
        for sentence, sent_start in sentences:
            sent_end = sent_start + len(sentence)
            # Find entities that fall inside this sentence span
            sentence_entities = [
                e for e in entities
                if e.get('offset', -1) >= sent_start and e.get('offset', -1) < sent_end
            ]
            
            if len(sentence_entities) >= 2:
                # Extract relationships between entities in the same sentence
                for i in range(len(sentence_entities)):
                    for j in range(i + 1, len(sentence_entities)):
                        entity1 = sentence_entities[i]
                        entity2 = sentence_entities[j]
                        
                        # Determine relationship type and strength
                        rel_type, confidence = self._analyze_relationship(
                            entity1['text'], entity2['text'], sentence,
                            entity1.get('offset'), entity2.get('offset')
                        )
                        
                        if confidence > 0.4:
                            key = tuple(sorted([entity1['text'], entity2['text']]) + [rel_type])
                            # Keep the strongest confidence for duplicate pairs
                            existing = next((r for r in relationships if r.get('key') == key), None)
                            if existing:
                                existing['confidence'] = max(existing['confidence'], confidence)
                            else:
                                relationships.append({
                                    'source': entity1['text'],
                                    'target': entity2['text'],
                                    'type': rel_type,
                                    'confidence': confidence,
                                    'context': sentence,
                                    'key': key
                                })
        
        # Remove helper keys before returning
        for r in relationships:
            r.pop('key', None)
        return relationships
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.
        
        Returns:
            Similarity score between 0 and 1
        """
        if not self.embeddings_model:
            # Fallback to simple word overlap
            return self._word_overlap_similarity(text1, text2)
        
        try:
            embeddings = self.embeddings_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return self._word_overlap_similarity(text1, text2)
    
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
        """
        Extract key phrases with importance scores.
        
        Returns:
            List of (phrase, score) tuples
        """
        sentences = self._split_sentences(text)
        phrases = []
        
        for sentence in sentences:
            # Extract noun phrases (simplified for Sinhala)
            words = sentence.split()
            
            # Look for meaningful multi-word phrases
            for i in range(len(words)):
                for j in range(i + 2, min(i + 6, len(words) + 1)):
                    phrase = ' '.join(words[i:j])
                    
                    if len(phrase) >= 10 and len(phrase) <= 80:
                        score = self._calculate_phrase_importance(phrase, text)
                        if score > 0.3:
                            phrases.append((phrase, score))
        
        # Deduplicate and sort
        phrase_dict = {}
        for phrase, score in phrases:
            if phrase not in phrase_dict or phrase_dict[phrase] < score:
                phrase_dict[phrase] = score
        
        sorted_phrases = sorted(phrase_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_phrases[:max_phrases]
    
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
        """Calculate importance score for a word."""
        # Check if it's a postposition or connector first (less important)
        if word in self.postpositions or word in self.connectors:
            return 0.0  # Return 0 to filter out completely
        
        score = 0.5  # Base score
        
        # Length factor (longer words often more important in Sinhala)
        if len(word) >= 5:
            score += 0.2
        if len(word) >= 8:
            score += 0.1
        
        # Position factor (beginning and end of sentences often important)
        if position < 2 or position >= total_words - 2:
            score += 0.1
        
        # Check if it's a verb indicator (context dependent)
        if any(verb in word for verb in self.verbs_indicators):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _classify_entity_type(self, word: str, context: str) -> str:
        """Classify the type of entity."""
        # Simple classification based on patterns
        if any(q in context for q in self.question_words):
            return 'query'
        elif any(verb in word for verb in self.verbs_indicators):
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
        
        # Check for specific relationship indicators
        if any(post in between for post in self.postpositions):
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
        """Remove duplicate entities, keeping highest importance."""
        entity_dict = {}
        
        for entity in entities:
            text = entity['text']
            if text not in entity_dict or entity_dict[text]['importance'] < entity['importance']:
                entity_dict[text] = entity
        
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
                similarity = self._word_overlap_similarity(
                    entity['text'], entities[j]['text']
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

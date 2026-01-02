"""
Advanced NLP Engine for Sinhala Text Processing
Uses lightweight models for intelligent node and relationship extraction.
"""

import re
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
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            # Extract proper nouns (capitalized words in Sinhala context)
            words = sentence.split()
            
            for i, word in enumerate(words):
                # Check if word is a potential entity
                if len(word) >= 3:
                    # Calculate importance based on position and context
                    importance = self._calculate_word_importance(word, sentence, i, len(words))
                    
                    if importance > 0.3:
                        entities.append({
                            'text': word,
                            'type': self._classify_entity_type(word, sentence),
                            'importance': importance,
                            'context': sentence
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
        sentences = self._split_sentences(text)
        
        entity_texts = {e['text'] for e in entities}
        
        for sentence in sentences:
            # Find entities in this sentence
            sentence_entities = [e for e in entities if e['text'] in sentence]
            
            if len(sentence_entities) >= 2:
                # Extract relationships between entities in the same sentence
                for i in range(len(sentence_entities)):
                    for j in range(i + 1, len(sentence_entities)):
                        entity1 = sentence_entities[i]
                        entity2 = sentence_entities[j]
                        
                        # Determine relationship type and strength
                        rel_type, confidence = self._analyze_relationship(
                            entity1['text'], entity2['text'], sentence
                        )
                        
                        if confidence > 0.4:
                            relationships.append({
                                'source': entity1['text'],
                                'target': entity2['text'],
                                'type': rel_type,
                                'confidence': confidence,
                                'context': sentence
                            })
        
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
        """Split text into sentences."""
        delimiters = ['.', '|', '?', '!', '।', ':', '\n']
        pattern = '|'.join(map(re.escape, delimiters))
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
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
    
    def _analyze_relationship(self, entity1: str, entity2: str, sentence: str) -> Tuple[str, float]:
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
        
        # Check for specific relationship indicators
        if any(post in between for post in self.postpositions):
            rel_type = 'semantic'
            confidence = 0.7
        elif any(conn in between for conn in self.connectors):
            rel_type = 'conjunction'
            confidence = 0.8
        elif any(verb in between for verb in self.verbs_indicators):
            rel_type = 'action'
            confidence = 0.75
        elif len(between.split()) <= 3:
            rel_type = 'close'
            confidence = 0.6
        
        return rel_type, confidence
    
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

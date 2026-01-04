"""
Sinhala Mind Map Generator Module
Processes Sinhala text and generates graph-ready mind map data.
"""

import re
import uuid
from typing import Dict, List, Any
from nlp_engine import SinhalaNLPEngine

class SinhalaMindMapGenerator:
    """Generates mind map structures from Sinhala text."""
    
    def __init__(self):
        """Initialize the mind map generator."""
        # Initialize NLP engine for label cleaning
        self.nlp_engine = SinhalaNLPEngine()
        
        # Common Sinhala sentence endings and punctuation
        self.sentence_delimiters = [':', '|', '?', '!', '।', '\n']
        self.topic_keywords = ['විෂය', 'මාතෘකා', 'පාඩම', 'අංශය', 'කොටස']
        
    def generate(self, text: str) -> Dict[str, Any]:
        """
        Generate a graph-ready mind map from Sinhala text.
        
        Args:
            text: Sinhala text to process
            
        Returns:
            Dictionary containing nodes and edges for graph visualization
        """
        if not text or not text.strip():
            return {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'total_nodes': 0,
                    'total_edges': 0
                }
            }
        
        # Extract main topics and subtopics
        nodes, edges = self._extract_structure(text)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'text_length': len(text)
            }
        }
    
    def _extract_structure(self, text: str) -> tuple:
        """
        Extract hierarchical structure from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (nodes, edges)
        """
        nodes = []
        edges = []
        
        # Create root node from first significant phrase
        lines = text.strip().split('\n')
        root_text = self._extract_main_topic(lines[0] if lines else text[:100])
        root_text = self.nlp_engine.clean_label(root_text)  # Remove helping words
        
        root_id = self._generate_id()
        nodes.append({
            'id': root_id,
            'label': root_text,
            'level': 0,
            'type': 'root',
            'size': 30
        })
        
        # Process remaining text to extract concepts
        paragraphs = self._split_into_paragraphs(text)
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip() or paragraph == root_text:
                continue
            
            # Split paragraph into sentences
            sentences = self._split_into_sentences(paragraph)
            
            # Create paragraph-level node
            para_id = self._generate_id()
            para_label = self._truncate_text(sentences[0] if sentences else paragraph, 50)
            para_label = self.nlp_engine.clean_label(para_label)  # Remove helping words
            
            nodes.append({
                'id': para_id,
                'label': para_label,
                'level': 1,
                'type': 'topic',
                'size': 20
            })
            
            # Connect to root
            edges.append({
                'id': self._generate_id(),
                'source': root_id,
                'target': para_id,
                'type': 'hierarchy'
            })
            
            # Process sentences as subtopics
            for j, sentence in enumerate(sentences[1:4]):  # Limit to 3 subtopics per topic
                if not sentence.strip():
                    continue
                
                sentence_id = self._generate_id()
                sentence_label = self._truncate_text(sentence, 40)
                sentence_label = self.nlp_engine.clean_label(sentence_label)  # Remove helping words
                
                nodes.append({
                    'id': sentence_id,
                    'label': sentence_label,
                    'level': 2,
                    'type': 'subtopic',
                    'size': 15
                })
                
                edges.append({
                    'id': self._generate_id(),
                    'source': para_id,
                    'target': sentence_id,
                    'type': 'hierarchy'
                })
                
                # Extract key phrases from sentence
                key_phrases = self._extract_key_phrases(sentence)
                
                for k, phrase in enumerate(key_phrases[:2]):  # Max 2 details per subtopic
                    phrase_id = self._generate_id()
                    cleaned_phrase = self.nlp_engine.clean_label(phrase)  # Remove helping words
                    
                    nodes.append({
                        'id': phrase_id,
                        'label': cleaned_phrase,
                        'level': 3,
                        'type': 'detail',
                        'size': 10
                    })
                    
                    edges.append({
                        'id': self._generate_id(),
                        'source': sentence_id,
                        'target': phrase_id,
                        'type': 'detail'
                    })
        
        return nodes, edges
    
    def _extract_main_topic(self, text: str) -> str:
        """Extract the main topic from text."""
        # Remove special characters and clean
        text = text.strip()
        
        # If text contains topic keywords, extract that part
        for keyword in self.topic_keywords:
            if keyword in text:
                parts = text.split(keyword)
                if len(parts) > 1:
                    return parts[1].strip()[:100]
        
        # Otherwise return cleaned first part
        return self._truncate_text(text, 80)
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or significant breaks
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using Sinhala delimiters."""
        # Replace Sinhala sentence endings with periods for consistent splitting
        for delimiter in self.sentence_delimiters:
            text = text.replace(delimiter, '.')
        
        # Split by period followed by space or newline
        sentences = re.split(r'\.\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
    
    def _extract_key_phrases(self, sentence: str) -> List[str]:
        """Extract key phrases from a sentence."""
        # Simple extraction based on length and content
        phrases = []
        
        # Split by commas or semicolons
        parts = re.split(r'[,;]', sentence)
        
        for part in parts:
            part = part.strip()
            # Keep meaningful phrases (between 10 and 50 characters)
            if 10 <= len(part) <= 50:
                phrases.append(part)
        
        # If no phrases found, try to extract noun phrases or meaningful chunks
        if not phrases and len(sentence) > 15:
            words = sentence.split()
            if len(words) >= 3:
                # Take first few words as a phrase
                phrases.append(' '.join(words[:min(5, len(words))]))
        
        return phrases
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length."""
        text = text.strip()
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + '...'
    
    def _generate_id(self) -> str:
        """Generate a unique ID for nodes and edges."""
        return str(uuid.uuid4())[:8]


# Alternative generator with more advanced NLP (when spaCy model is available)
class AdvancedSinhalaMindMapGenerator(SinhalaMindMapGenerator):
    """
    Advanced mind map generator using NLP techniques.
    Requires spaCy or similar NLP library with Sinhala support.
    """
    
    def __init__(self):
        super().__init__()
        # Future: Load spaCy model for Sinhala
        # self.nlp = spacy.load('si_core_news_sm')
        pass
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Future implementation with NLP
        # doc = self.nlp(text)
        # return [ent.text for ent in doc.ents]
        return []
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords from text."""
        # Future implementation with TF-IDF or similar
        return []

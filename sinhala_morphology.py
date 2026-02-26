"""
Sinhala Morphology Handler

Handles compound word splitting, inflection normalization, and root extraction
for improved entity extraction in Sinhala text.
"""

import re
import logging
from typing import List, Tuple, Set, Optional

logger = logging.getLogger(__name__)


class SinhalaMorphologyHandler:
    """
    Advanced Sinhala morphology processing for entity extraction.
    
    Features:
    - Compound word splitting
    - Inflection removal/normalization
    - Root word extraction
    - Sandhi resolution (basic)
    """
    
    def __init__(self):
        # Common inflectional suffixes (case, number, tense)
        self.case_suffixes = [
            'ගෙන්', 'ගේ', 'ට', 'ගෙන', 'වලින්', 'වල', 'යින්', 
            'තුළින්', 'තුළ', 'ගින්', 'ගින', 'ගත', 'කින්'
        ]
        
        self.verb_suffixes = [
            'නවා', 'ෙනවා', 'ිනවා', 'ීවා', 'ූවා', 'ාවා',
            'ෙයි', 'යි', 'ෙනු', 'ෙන', 'ෙමු', 'ෙහෙ', 'න්න',
            'ල', 'ලා', 'මින්', 'මි', 'මු', 'ත්', 'ද',
            'නා', 'නු', 'මි', 'මු', 'ති', 'න්නේ', 'නවාය'
        ]
        
        # Plural markers
        self.plural_suffixes = [
            'ජනයා', 'ජනයෝ', 'වරු', 'වරුන්', 'වරුන්',  'හු', 'ඝන',
            'ගණය', 'කාරයෝ', 'කාරයින්', 'යෝ', 'යන්', 'න්'
        ]
        
        # Common compound connectors
        self.compound_connectors = ['හා', 'සහ', 'හි', 'ගේ', 'යායි']
        
        # Sandhi consonant replacements (simplified)
        self.sandhi_patterns = [
            (r'එකක්', 'එක'),
            (r'යක්', 'ය'),
            (r'වක්', 'ව'),
            (r'මක්', 'ම'),
        ]
        
    def normalize_word(self, word: str) -> str:
        """
        Normalize a Sinhala word by removing inflections and returning root form.
        
        Args:
            word: Sinhala word
            
        Returns:
            Normalized root form
        """
        if not word or len(word) < 3:
            return word
            
        original = word
        
        # Try removing case suffixes
        for suffix in sorted(self.case_suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                logger.debug(f"Removed case suffix '{suffix}' from '{original}' → '{word}'")
                break
        
        # Try removing plural markers
        for suffix in sorted(self.plural_suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                logger.debug(f"Removed plural suffix '{suffix}' from '{original}' → '{word}'")
                break
        
        # Try removing verb suffixes
        for suffix in sorted(self.verb_suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)]
                logger.debug(f"Removed verb suffix '{suffix}' from '{original}' → '{word}'")
                break
        
        # Apply sandhi resolution
        for pattern, replacement in self.sandhi_patterns:
            word = re.sub(pattern, replacement, word)
        
        return word
    
    def split_compound(self, compound: str) -> List[str]:
        """
        Split a Sinhala compound word into components.
        
        Args:
            compound: Compound word
            
        Returns:
            List of component words
        """
        # Check for explicit compound connectors
        for connector in self.compound_connectors:
            if connector in compound:
                parts = compound.split(connector)
                return [p.strip() for p in parts if p.strip()]
        
        # Heuristic splitting based on length and common patterns
        # This is simplified - real compound splitting needs more sophisticated rules
        if len(compound) > 15:
            # Try splitting at common boundaries
            for i in range(4, len(compound) - 4):
                prefix = compound[:i]
                suffix = compound[i:]
                if self._is_likely_word_boundary(prefix, suffix):
                    return [prefix, suffix]
        
        return [compound]
    
    def _is_likely_word_boundary(self, prefix: str, suffix: str) -> bool:
        """Check if the split point is likely a word boundary."""
        # Simple heuristics
        if len(prefix) < 3 or len(suffix) < 3:
            return False
        
        # Check if prefix ends with a common noun suffix
        noun_endings = ['ය', 'ව', 'ම', 'ිය', 'ීය', 'ාව', 'ීම']
        if any(prefix.endswith(e) for e in noun_endings):
            # Check if suffix starts with a consonant
            if suffix and self._is_consonant(suffix[0]):
                return True
        
        return False
    
    def _is_consonant(self, char: str) -> bool:
        """Check if a character is a Sinhala consonant."""
        consonant_range = range(0x0D9A, 0x0DC6 + 1)  # Sinhala consonants
        return ord(char) in consonant_range
    
    def extract_root(self, word: str) -> str:
        """
        Extract the core root from a word by removing all affixes.
        
        Args:
            word: Sinhala word
            
        Returns:
            Root form
        """
        # First normalize
        normalized = self.normalize_word(word)
        
        # Additional aggressive suffix removal for root extraction
        aggressive_suffixes = ['ටා', 'මයි', 'ගිය', 'ගත', 'වත්', 'යේ', 'යෙන්']
        for suffix in aggressive_suffixes:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                normalized = normalized[:-len(suffix)]
        
        return normalized
    
    def is_inflected_form(self, word1: str, word2: str) -> bool:
        """
        Check if two words are inflected forms of the same root.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if they share the same root
        """
        root1 = self.extract_root(word1)
        root2 = self.extract_root(word2)
        
        return root1 == root2 and root1 != word1 and root2 != word2
    
    def get_citation_form(self, word: str) -> str:
        """
        Get the dictionary/citation form of a word (nominative singular for nouns).
        
        Args:
            word: Inflected word
            
        Returns:
            Citation form
        """
        # For now, this is similar to extract_root
        # Could be enhanced with more sophisticated rules
        return self.extract_root(word)


# Singleton instance
_morphology_handler: Optional[SinhalaMorphologyHandler] = None


def get_morphology_handler() -> SinhalaMorphologyHandler:
    """Get or create the singleton morphology handler."""
    global _morphology_handler
    if _morphology_handler is None:
        _morphology_handler = SinhalaMorphologyHandler()
    return _morphology_handler


def normalize_sinhala_word(word: str) -> str:
    """Convenience function to normalize a Sinhala word."""
    return get_morphology_handler().normalize_word(word)


def split_compound_word(word: str) -> List[str]:
    """Convenience function to split a compound word."""
    return get_morphology_handler().split_compound(word)

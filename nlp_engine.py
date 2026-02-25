"""
Advanced NLP Engine for Sinhala Text Processing
Uses lightweight models for intelligent node and relationship extraction.
"""

import re
import hashlib
import math
import os
import sys
import unicodedata
from typing import List, Dict, Tuple, Set, Any
import numpy as np
from collections import defaultdict, Counter
import logging
from config import Config
from hybrid_node_extractor import HybridNodeExtractor
from relation_classifier import RelationClassifier

logger = logging.getLogger(__name__)


class CachedEmbeddingModel:
    """Thin LRU-style cache wrapper around a sentence-transformer model."""

    def __init__(self, model: Any, max_cache_size: int = 4096):
        self.model = model
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []

    def _touch(self, key: str) -> None:
        if key in self._cache_order:
            self._cache_order.remove(key)
        self._cache_order.append(key)
        while len(self._cache_order) > self.max_cache_size:
            evict_key = self._cache_order.pop(0)
            self._cache.pop(evict_key, None)

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        result: List[np.ndarray] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for idx, text in enumerate(texts):
            key = text or ""
            if key in self._cache:
                result.append(self._cache[key])
                self._touch(key)
            else:
                result.append(None)
                uncached_indices.append(idx)
                uncached_texts.append(key)

        if uncached_texts:
            new_embs = self.model.encode(uncached_texts, **kwargs)
            new_embs = np.asarray(new_embs)
            for idx, emb in zip(uncached_indices, new_embs):
                key = texts[idx] or ""
                vec = np.asarray(emb, dtype=np.float32)
                self._cache[key] = vec
                self._touch(key)
                result[idx] = vec

        out = np.vstack(result) if result else np.array([], dtype=np.float32)
        return out[0] if single_input else out


class TransformerMeanPoolingEmbedder:
    """Fallback embedding backend using transformers + torch mean pooling."""

    def __init__(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        batch_size = int(kwargs.get('batch_size', 32))
        convert_to_numpy = kwargs.get('convert_to_numpy', True)
        all_embs: List[np.ndarray] = []

        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt',
                )
                outputs = self.model(**tokens)
                token_embeddings = outputs.last_hidden_state
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                masked = token_embeddings * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                emb = summed / counts
                emb = self.torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embs.append(emb.detach().cpu().numpy().astype(np.float32))

        result = np.vstack(all_embs) if all_embs else np.array([], dtype=np.float32)
        if convert_to_numpy:
            return result[0] if single_input else result
        return result[0] if single_input else result


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
        
        # Lightweight Sinhala synonym lexicon (WordNet-style fallback)
        self.sinhala_synonyms: Dict[str, Set[str]] = self._load_sinhala_synonyms()

        # Initialize sentence transformer for semantic analysis
        # Optional bypass for environments where torch/transformers is unavailable
        disable_embeddings = os.getenv('DISABLE_SENTENCE_TRANSFORMERS', 'false').lower() == 'true'
        if disable_embeddings:
            logger.info("Sentence-transformers disabled via DISABLE_SENTENCE_TRANSFORMERS=true")
        else:
            try:
                os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
                from sentence_transformers import SentenceTransformer
                # Use a lightweight multilingual model that supports Sinhala
                base_model = SentenceTransformer(self.embeddings_model_name, device='cpu')
                self.embeddings_model = CachedEmbeddingModel(base_model)
                logger.info("Loaded multilingual embeddings model: %s", self.embeddings_model_name)
            except Exception as e:
                logger.warning(
                    "Could not load embeddings model %s on Python %s: %s",
                    self.embeddings_model_name,
                    sys.version.split()[0],
                    e,
                )
                # Compatibility fallback for Python 3.12: use raw transformers + torch
                try:
                    fallback_model = TransformerMeanPoolingEmbedder(self.embeddings_model_name)
                    self.embeddings_model = CachedEmbeddingModel(fallback_model)
                    logger.info(
                        "Loaded fallback embeddings backend (transformers+torch): %s",
                        self.embeddings_model_name,
                    )
                except Exception as e2:
                    logger.warning(
                        "Fallback transformers embedding load failed for %s: %s",
                        self.embeddings_model_name,
                        e2,
                    )
        
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
        
        # ENHANCED Sinhala stop words (helping words that should not appear in nodes)
        # Expanded from 114 to 180+ terms for better filtering
        self.stop_words = set([
            # Pronouns
            'මම', 'ඔබ', 'ඔහු', 'ඇය', 'අපි', 'ඔවුන්', 'මා', 'අප', 'ඔබට', 'මට',
            'ඔහුට', 'ඇයට', 'අපට', 'ඔවුන්ට', 'මගේ', 'ඔබේ', 'ඔහුගේ', 'ඇයගේ',
            'අපගේ', 'ඔවුන්ගේ', 'මාගේ', 'තමා', 'තමන්', 'තමන්ගේ',
            
            # Articles and determiners
            'එක්', 'එක', 'මේ', 'මෙම', 'ඒ', 'ඔය', 'ඕ', 'කිසිවක්', 'සියලු', 'සියළු',
            'අනෙක්', 'තවත්', 'තව', 'හැම', 'හැමඳා', 'එම', 'මෙසේ', 'එලෙස',
            'අනෙක', 'මෙලෙස', 'ඒ පරිදි', 'මෙ පරිදි', 'ඒ ආකාරයට',
            
            # Common helping verbs (EXPANDED)
            'වන', 'වන්නේ', 'වේ', 'වී', 'වෙයි', 'වුණා', 'වෙන්නේ', 'ඇති', 'ඇත', 'ඇත්තේ',
            'වෙන', 'වූ', 'වු', 'වුණු', 'වෙච්', 'වෙලා', 'වෙන්න', 'වෙන්නේ', 'වෙන්නෙ',
            'වෙච්ච', 'වෙද්දී', 'වෙද්දි', 'වෙනකොට', 'වෙනව', 'වෙනවා',
            'හැක', 'හැකි', 'හැකිය', 'හැකියි', 'හැකියාව', 'පුළුවන්', 'පුළුවන',
            'අවශ්‍ය', 'තිබේ', 'තිබෙන', 'තිබුණ', 'තිබුණා', 'තිබෙන්නේ',
            'ඇත', 'නැත', 'නැති', 'නැතිව', 'නැතුව',
            
            # Postpositions (EXPANDED)
            'සඳහා', 'විසින්', 'ගැන', 'පිළිබඳව', 'අතර', 'තුළ', 'හි', 'සමග', 'සමඟ', 'සිට', 'දක්වා', 'මගින්',
            'වෙත', 'පැත්තේ', 'පැත්තෙන්', 'කරා', 'දී', 'දීම', 'බවට', 'වශයෙන්',
            'අනුව', 'අනුකූලව', 'අනුසාරව', 'අනුගතව', 'අනුකූලව', 'අනුරූප',
            'මූලික', 'මත', 'මතින්', 'ආශ්‍රිතව', 'පාදක', 'පදනම්',
            
            # Conjunctions (EXPANDED)
            'සහ', 'හා', 'මෙන්ම', 'ද', 'වගේම', 'පමණක්', 'නමුත්', 'කෙසේ', 'හෝ',
            'වුවද', 'වුවත්', 'මෙන්', 'සේ', 'වන්', 'හෝ', 'හෝද', 'නොහොත්',
            'එනමුත්', 'එහෙත්', 'එහෙත', 'එවිට', 'එවිටදී', 'එවිට', 'එකින්',
            'එහෙත්', 'ඉන්පසු', 'ඉන්', 'පසු', 'පසුව', 'පසුකාලීනව',
            
            # Auxiliary words (EXPANDED)
            'ඉතා', 'වඩා', 'මෙන්', 'තරම්', 'පමණ', 'ලෙස', 'ලෙසින්', 'යනාදී', 'ආදිය',
            'විශේෂයෙන්', 'විශේෂයෙන්ම', 'මූලික', 'මූලිකව', 'ප්‍රධාන',
            'ප්‍රධාන', 'මෙම', 'මෙසේ', 'එලෙස', 'මේ', 'එය', 'එ', 'ඒ',
            'ඉතාමත්', 'ඉතාමත්ම', 'අතිශයින්', 'අතිශයින්ම',
            
            # Common particles
            'ද', 'ය', 'ක්', 'යි', 'නේ', 'නේද', 'ඩ', 'නම්', 'වනම්', 'නම',
            'ඳ', 'ක', 'ද', 'යි', 'නෙ', 'වෙයි', 'වෙයිද', 'නෙද',
            
            # Question particles (EXPANDED)
            'කවර', 'කොහොමද', 'කුමක්', 'ඇයි', 'මොකද', 'කවදා', 'කොතැනද', 'කොහෙද',
            'කවරේ', 'කවර', 'කවරද', 'කවරදැයි', 'කවදාද', 'කොහෙන්', 'කොහෙද',
            'කෙසේ', 'කෙසේද', 'කෙසේදැයි', 'මොනවා', 'මොනවද', 'මොන', 'මොකක්',
            
            # Modal particles (EXPANDED)
            'බව', 'නිසා', 'නම්', 'යැයි', 'කියා', 'කියලා', 'කියන', 'කී',
            'දැයි', 'දයි', 'යයි', 'යි', 'වෙයි', 'යැයිද', 'දැයිද',
            
            # Short common words (EXPANDED)
            'ට', 'ටත්', 'යන', 'දී', 'ගේ', 'වල', 'ය', 'යි', 'ගෙන්', 'කර',
            'වන', 'තුළ', 'හා', 'සහ', 'සහිත', 'සහිතව', 'සමග', 'සමඟ',
            
            # Temporal/Aspect markers
            'විට', 'වී', 'වූ', 'වන', 'වෙමින්', 'වෙමින', 'ලද', 'ලබන', 'ලබා',
            'තිබෙන', 'තිබූ', 'තිබෙමින්', 'තිබෙමින', 'පවතින', 'පවතින්නා',
            
            # Numbers and quantifiers (that are not concepts)
            'එක', 'දෙක', 'තුන', 'හතර', 'පස', 'හය', 'හත', 'අට', 'නමය',
            'දහය', 'සියය', 'දහස', 'ලක්ෂය', 'මිලියනය',
            'බොහෝ', 'සෑම', 'සෑම', 'සෑමක්', 'සෑමක', 'සියලු', 'සියලුම',
            'සියල්ල', 'සියල්', 'ඇතැම්', 'ඇතැම', 'ඇතැමක්', 'කිහිපයක්',
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
            return self._fallback_semantic_similarity(text1, text2)
        
        try:
            embeddings = self.embeddings_model.encode(
                [text1, text2],
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return self._fallback_semantic_similarity(text1, text2)
    
    def cluster_concepts(self, entities: List[Dict], threshold: float = 0.6) -> List[List[Dict]]:
        """
        Cluster similar entities together based on semantic similarity.
        
        Returns:
            List of entity clusters
        """
        if not entities:
            return []
        
        if not self.embeddings_model:
            # Better fallback: TF-IDF + LSA semantic clustering
            return self._tfidf_lsa_cluster(entities, threshold=max(0.35, threshold - 0.1))
        
        try:
            # Extract texts for embedding
            texts = [e['text'] for e in entities]
            embeddings = self.embeddings_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=64,
            )

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

    def _synonym_overlap_similarity(self, text1: str, text2: str) -> float:
        """Compute overlap after expanding tokens via Sinhala synonym sets."""
        tokens1 = {self._normalize_term(t) for t in self._tokenize(text1)}
        tokens2 = {self._normalize_term(t) for t in self._tokenize(text2)}
        tokens1 = {t for t in tokens1 if t}
        tokens2 = {t for t in tokens2 if t}

        if not tokens1 or not tokens2:
            return 0.0

        def expand(tokens: Set[str]) -> Set[str]:
            expanded = set(tokens)
            for token in tokens:
                for syn in self.sinhala_synonyms.get(token, set()):
                    expanded.add(self._normalize_term(syn))
            return {t for t in expanded if t}

        expanded1 = expand(tokens1)
        expanded2 = expand(tokens2)
        union = len(expanded1 | expanded2)
        if union == 0:
            return 0.0
        return len(expanded1 & expanded2) / union

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

    def _fallback_semantic_similarity(self, text1: str, text2: str) -> float:
        """Weighted fallback similarity: synonym overlap + TF-IDF/LSA + lexical + n-gram."""
        synonym_sim = self._synonym_overlap_similarity(text1, text2)
        lexical_sim = self._word_overlap_similarity(text1, text2)
        char_sim = self._char_ngram_similarity(text1, text2)
        lsa_sim = self._tfidf_lsa_pair_similarity(text1, text2)
        score = (0.35 * synonym_sim) + (0.35 * lsa_sim) + (0.2 * lexical_sim) + (0.1 * char_sim)
        return float(max(0.0, min(1.0, score)))

    def _tfidf_lsa_pair_similarity(self, text1: str, text2: str) -> float:
        """Compute pair similarity using lightweight TF-IDF + truncated SVD (LSA)."""
        docs = [text1 or "", text2 or ""]
        matrix, _ = self._build_tfidf_matrix(docs)
        if matrix.shape[1] == 0:
            return max(self._word_overlap_similarity(text1, text2), self._char_ngram_similarity(text1, text2))

        # LSA projection with small rank for stability on tiny corpora
        rank = min(2, matrix.shape[0], matrix.shape[1])
        if rank >= 1:
            try:
                u, s, vt = np.linalg.svd(matrix, full_matrices=False)
                lsa = u[:, :rank] * s[:rank]
                a, b = lsa[0], lsa[1]
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
                return float(np.dot(a, b) / denom)
            except Exception:
                pass

        a, b = matrix[0], matrix[1]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _build_tfidf_matrix(self, docs: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Build a lightweight TF-IDF matrix without external dependencies."""
        tokenized_docs: List[List[str]] = []
        df_counter: Counter = Counter()

        for doc in docs:
            tokens = [self._normalize_term(t) for t in self._tokenize(doc)]
            tokens = [t for t in tokens if t and t not in self.stop_words and len(t) >= 2]
            tokenized_docs.append(tokens)
            for token in set(tokens):
                df_counter[token] += 1

        vocab = sorted(df_counter.keys())
        if not vocab:
            return np.zeros((len(docs), 0), dtype=np.float32), []

        token_to_idx = {t: i for i, t in enumerate(vocab)}
        matrix = np.zeros((len(docs), len(vocab)), dtype=np.float32)
        num_docs = max(1, len(docs))

        for doc_idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf_counter = Counter(tokens)
            total = len(tokens)
            for token, tf in tf_counter.items():
                col = token_to_idx[token]
                tf_val = tf / total
                idf = math.log((1 + num_docs) / (1 + df_counter[token])) + 1.0
                matrix[doc_idx, col] = tf_val * idf

        return matrix, vocab

    def _tfidf_lsa_cluster(self, entities: List[Dict], threshold: float = 0.5) -> List[List[Dict]]:
        """Fallback clustering using TF-IDF + LSA similarity matrix."""
        if not entities:
            return []

        texts = [e.get('text', '') for e in entities]
        tfidf, _ = self._build_tfidf_matrix(texts)
        if tfidf.shape[1] == 0:
            return self._simple_cluster(entities)

        rank = min(16, tfidf.shape[0], tfidf.shape[1])
        if rank >= 2:
            try:
                u, s, vt = np.linalg.svd(tfidf, full_matrices=False)
                vectors = u[:, :rank] * s[:rank]
            except Exception:
                vectors = tfidf
        else:
            vectors = tfidf

        clusters: List[List[Dict]] = []
        used: Set[int] = set()

        for i, entity in enumerate(entities):
            if i in used:
                continue
            cluster = [entity]
            used.add(i)
            for j in range(i + 1, len(entities)):
                if j in used:
                    continue
                a = vectors[i]
                b = vectors[j]
                sim = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
                syn_sim = self._synonym_overlap_similarity(texts[i], texts[j])
                mixed = 0.75 * sim + 0.25 * syn_sim
                if mixed >= threshold:
                    cluster.append(entities[j])
                    used.add(j)
            clusters.append(cluster)

        return clusters

    def _load_sinhala_synonyms(self) -> Dict[str, Set[str]]:
        """Load Sinhala synonym mapping (WordNet-style) from env JSON path or defaults."""
        synonym_map: Dict[str, Set[str]] = {
            'ගුරු': {'ආචාර්ය', 'ශික්ෂක'},
            'ශිෂ්‍ය': {'සිසුවා', 'දරුවා'},
            'අධ්‍යාපනය': {'ඉගෙනීම', 'ශික්ෂණය'},
            'සෞඛ්‍යය': {'ආරෝග්‍යය', 'නිරෝගීභාවය'},
            'රෝගය': {'ලෙඩ', 'අසනීපය'},
            'හේතුව': {'කාරණය', 'මූලය'},
            'ප්‍රතිඵලය': {'ප්‍රතිඵල', 'ඵලය'},
            'සමාජය': {'සමාජ', 'ප්‍රජාව'},
            'පරිසරය': {'වාතාවරණය', 'පරිසර'},
            'දැනුම': {'ඥානය', 'විද්‍යාව'},
        }

        custom_path = os.getenv('SINHALA_WORDNET_PATH', '').strip()
        if not custom_path:
            return synonym_map

        try:
            import json
            with open(custom_path, encoding='utf-8') as fp:
                payload = json.load(fp)
            if isinstance(payload, dict):
                for key, values in payload.items():
                    if not isinstance(values, list):
                        continue
                    key_norm = self._normalize_term(str(key))
                    if not key_norm:
                        continue
                    syns = {self._normalize_term(str(v)) for v in values}
                    syns = {s for s in syns if s}
                    if key_norm not in synonym_map:
                        synonym_map[key_norm] = set()
                    synonym_map[key_norm].update(syns)
        except Exception as e:
            logger.warning("Failed to load Sinhala WordNet synonyms from %s: %s", custom_path, e)

        return synonym_map
    
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

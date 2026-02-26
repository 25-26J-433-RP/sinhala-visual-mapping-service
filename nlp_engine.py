"""
Advanced NLP Engine for Sinhala Text Processing
Uses lightweight models for intelligent node and relationship extraction.
"""

import re
import csv
import hashlib
import math
import os
import sys
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
from collections import defaultdict, Counter
import logging
from config import Config
from hybrid_node_extractor import HybridNodeExtractor
from relation_classifier import RelationClassifier
from sinhala_morphology import get_morphology_handler

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

        # Sinhala morphology / stemming / lemmatization support
        self.morphology = get_morphology_handler()
        
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

        # Context anchors for word-sense disambiguation in essays (base lexicon)
        self.base_disambiguation_domains: Dict[str, Set[str]] = {
            'education': {'පාසල්', 'ශිෂ්‍ය', 'ගුරු', 'ඉගෙනීම', 'විෂය', 'රචනය', 'විභාග'},
            'science': {'විද්‍යාව', 'පර්යේෂණ', 'පරීක්ෂණ', 'සංකල්ප', 'රසායන', 'භෞතික'},
            'society': {'සමාජය', 'ජනතාව', 'ප්‍රජාව', 'සංස්කෘතිය', 'සහභාගීත්වය'},
            'environment': {'පරිසරය', 'වනය', 'ජලය', 'දේශගුණය', 'දූෂණය', 'සුරැකීම'},
            'health': {'සෞඛ්‍යය', 'රෝගය', 'ප්‍රතිකාර', 'වෛද්‍ය', 'ආහාර', 'ව්‍යායාම'},
        }
        self.disambiguation_domains: Dict[str, Set[str]] = {
            k: set(v) for k, v in self.base_disambiguation_domains.items()
        }
        self.ambiguous_term_domains: Dict[str, Dict[str, float]] = {}

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
        self.comparison_markers = ['වඩා', 'සසඳා', 'සසඳන', 'vs', 'VS', 'වර්සස්', 'අතර වෙනස', 'සමානතා']
        self.argument_markers = ['පළමුව', 'දෙවනුව', 'තෙවනුව', 'එක් පාර්ශවයෙන්', 'අනෙක් පාර්ශවයෙන්', 'නමුත්', 'එහෙත්', 'එයට ප්‍රතිවිරුද්ධව']
        self.process_markers = ['පළමුව', 'ඊළඟට', 'පසුව', 'අවසානයේ', 'පියවර', 'ක්‍රියාවලිය', 'අදියර']
        self.enumeration_lead_markers = ['ලෙස', 'වශයෙන්', 'ඇතුළුව', 'ආදිය', 'යනාදී', 'වැනි']
        self.numbering_pattern = re.compile(r'(^|\s)(\d+[\)\.]|[a-zA-Z][\)\.]|[\u0D85-\u0DC6][\)\.])\s*')
        
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

        # Tune domain anchors with actual essay corpus frequencies (requires stop_words)
        self.disambiguation_domains = self._load_corpus_tuned_disambiguation_domains(
            self.base_disambiguation_domains
        )
        self.ambiguous_term_domains = self._build_ambiguous_term_domains(
            self.disambiguation_domains
        )

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
        """Hybrid enumeration extractor with boundary scoring, comparison and nested list support."""
        enumerations: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, Tuple[str, ...], str]] = set()
        sentences = self._split_sentences_with_spans(text)

        for sentence, _ in sentences:
            if not sentence or len(sentence) < 10:
                continue

            # 1) Comparison/contrast structures (X vs Y, X සහ Y අතර වෙනස)
            comparison_enum = self._extract_comparison_structure(sentence)
            has_comparison = comparison_enum is not None
            if comparison_enum:
                key = self._enumeration_signature(
                    comparison_enum.get('head'), comparison_enum.get('items', []), comparison_enum.get('relation', '')
                )
                if key not in seen:
                    seen.add(key)
                    enumerations.append(comparison_enum)

            # 2) Argument/discourse structures in student essays
            argument_enum = self._extract_argument_structure(sentence)
            if argument_enum:
                key = self._enumeration_signature(
                    argument_enum.get('head'), argument_enum.get('items', []), argument_enum.get('relation', '')
                )
                if key not in seen:
                    seen.add(key)
                    enumerations.append(argument_enum)

            # 3) Explicit list markers and process-step enumerations
            explicit_candidates = []
            if any(marker in sentence for marker in self.sequence_markers):
                list_text = ''
                for marker in self.enumeration_lead_markers:
                    list_text = self._extract_list_after_marker(sentence, marker)
                    if list_text:
                        break
                if not list_text:
                    list_text = sentence
                explicit_candidates.append((
                    self._infer_head_phrase(sentence, self.sequence_markers) or 'අදියර',
                    list_text,
                    'sequence'
                ))

            if 'වැනි' in sentence or any(marker in sentence for marker in self.example_markers):
                before, after = sentence.split('වැනි', 1) if 'වැනි' in sentence else (sentence, '')
                head = self._first_token(after) or self._infer_head_phrase(sentence, ['භාෂා', 'ක්‍රමවේද', 'මූලධර්ම'])
                explicit_candidates.append((head, before, 'example'))

            if any(marker in sentence for marker in self.importance_markers) and not has_comparison:
                explicit_candidates.append((None, sentence, 'requires'))

            # Nested enumerations like "X: a, b; Y: c, d"
            if ':' in sentence and ('(' in sentence or ';' in sentence):
                nested_list_text = sentence.split(':', 1)[1] if ':' in sentence else sentence
                explicit_candidates.append((
                    self._infer_head_phrase(sentence, ['වර්ග', 'වර්ගීකරණය', 'ප්‍රභේද', 'අංග']) or 'වර්ග',
                    nested_list_text,
                    'group'
                ))

            # Implicit lists (comma/connector heavy, process cues, numbered cues)
            implicit_items, implicit_nested = self._extract_list_items_with_nested(sentence)
            has_list_cues = any(sep in sentence for sep in [',', ';', ' සහ ', ' හා ', ' මෙන්ම '])
            if has_list_cues and len(implicit_items) >= 3:
                implicit_relation = 'sequence' if any(m in sentence for m in self.process_markers) else 'group'
                implicit_head = self._infer_head_phrase(sentence, self.sequence_markers + ['වර්ග', 'අංග', 'අංශ', 'මූලධර්ම'])
                explicit_candidates.append((implicit_head or None, sentence, implicit_relation))

            for head, list_text, relation in explicit_candidates:
                items, nested_map = self._extract_list_items_with_nested(list_text)
                if len(items) < 2:
                    continue
                boundary_score = self._score_list_boundary(sentence, items, relation)
                threshold = 0.5 if relation in {'comparison', 'argument'} else 0.56
                if nested_map and relation == 'group':
                    threshold = 0.45
                if boundary_score < threshold:
                    continue
                key = self._enumeration_signature(head, items, relation)
                if key in seen:
                    continue
                seen.add(key)
                enum_payload = {
                    'head': head,
                    'items': items,
                    'relation': relation,
                    'confidence': round(boundary_score, 4),
                }
                if nested_map:
                    enum_payload['nested_items'] = nested_map
                elif relation in {'group', 'sequence'} and implicit_nested:
                    enum_payload['nested_items'] = implicit_nested
                enumerations.append(enum_payload)

        return self._deduplicate_enumeration_candidates(enumerations)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (legacy helper)."""
        return [s for s, _ in self._split_sentences_with_spans(text)]

    def _split_sentences_with_spans(self, text: str) -> List[Tuple[str, int]]:
        """Split text into sentences and return (sentence, start_index)."""
        delimiters = ['.', '|', '?', '!', '।', '\n']
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
        text = self._normalize_unicode(text)
        if self.sinling_tokenizer:
            try:
                tokens = self.sinling_tokenizer.tokenize(text)
                base_tokens = [t for t in tokens if t and t.strip()]
                return self._expand_sandhi_tokens(base_tokens)
            except Exception:
                pass
        # Remove zero-width characters then extract Sinhala and Latin tokens
        text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
        base_tokens = self.word_pattern.findall(text)
        return self._expand_sandhi_tokens(base_tokens)

    def _expand_sandhi_tokens(self, tokens: List[str]) -> List[str]:
        """Expand likely compound/sandhi tokens into component roots (without losing original token)."""
        expanded: List[str] = []
        for token in tokens:
            norm_token = self._normalize_term(token)
            if not norm_token:
                continue
            expanded.append(norm_token)
            if len(norm_token) >= 6:
                parts = self.morphology.split_compound(norm_token)
                if len(parts) > 1:
                    for part in parts:
                        part_norm = self._normalize_term(part)
                        if part_norm and part_norm != norm_token:
                            expanded.append(part_norm)
        return expanded

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
        candidates = self._split_list_candidates(text)
        items = []
        for cand in candidates:
            cand = re.sub(r'[\[\]{}"\'\u201c\u201d]', ' ', cand)
            cand = self.numbering_pattern.sub(' ', cand).strip()
            label = self.clean_label(cand)
            if label and len(label) >= 2 and len(label) <= 64 and label not in items:
                items.append(label)
        return items

    def _split_list_candidates(self, text: str) -> List[str]:
        """Split candidate list items while preserving nested parenthesis spans."""
        if not text:
            return []
        raw = self._normalize_unicode(text)
        depth = 0
        buff: List[str] = []
        parts: List[str] = []
        i = 0

        while i < len(raw):
            ch = raw[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)

            if depth == 0:
                matched_connector = None
                for connector in [' සහ ', ' හා ', ' මෙන්ම ']:
                    if raw.startswith(connector, i):
                        matched_connector = connector
                        break
                if matched_connector:
                    segment = ''.join(buff).strip()
                    if segment:
                        parts.append(segment)
                    buff = []
                    i += len(matched_connector)
                    continue

                if ch in [',', ';', '|', '/']:
                    segment = ''.join(buff).strip()
                    if segment:
                        parts.append(segment)
                    buff = []
                    i += 1
                    continue

            buff.append(ch)
            i += 1

        tail = ''.join(buff).strip()
        if tail:
            parts.append(tail)
        return parts

    def _extract_list_items_with_nested(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Extract list items and nested child lists (e.g., A: a1, a2)."""
        if not text:
            return [], {}

        nested_map: Dict[str, List[str]] = {}
        items: List[str] = []
        candidates = self._split_list_candidates(text)
        if not candidates:
            return [], {}

        for cand in candidates:
            cand = self.numbering_pattern.sub(' ', cand).strip()
            parent = cand
            children: List[str] = []

            if ':' in cand:
                left, right = cand.split(':', 1)
                parent = left.strip()
                children = self._extract_list_items(right)
            elif ' - ' in cand:
                left, right = cand.split(' - ', 1)
                parent = left.strip()
                children = self._extract_list_items(right)
            elif '(' in cand and ')' in cand:
                left = cand.split('(', 1)[0].strip()
                inner = cand.split('(', 1)[1].rsplit(')', 1)[0]
                parent = left or cand
                children = self._extract_list_items(inner)

            parent_label = self.clean_label(parent)
            if not parent_label or len(parent_label) < 2:
                continue
            if parent_label not in items:
                items.append(parent_label)

            clean_children = []
            for child in children:
                c = self.clean_label(child)
                if c and c != parent_label and c not in clean_children:
                    clean_children.append(c)
            if clean_children:
                nested_map[parent_label] = clean_children

        return items, nested_map

    def _deduplicate_enumeration_candidates(self, enumerations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reduce overlaps across extraction paths and keep strongest candidate per item signature."""
        if not enumerations:
            return []

        relation_priority = {
            'comparison': 6,
            'argument': 5,
            'sequence': 4,
            'requires': 3,
            'example': 2,
            'group': 1,
        }
        best_by_items: Dict[Tuple[str, ...], Dict[str, Any]] = {}

        for enum in enumerations:
            items = [self._normalize_term(i) for i in enum.get('items', []) if i]
            if len(items) < 2:
                continue
            key = tuple(items)
            score = float(enum.get('confidence', 0.5))
            score += 0.03 * relation_priority.get(enum.get('relation', ''), 0)
            existing = best_by_items.get(key)
            if not existing:
                best_by_items[key] = dict(enum)
                best_by_items[key]['_rank'] = score
                continue

            if score > float(existing.get('_rank', 0.0)):
                repl = dict(enum)
                repl['_rank'] = score
                best_by_items[key] = repl

        deduped = []
        for enum in best_by_items.values():
            enum.pop('_rank', None)
            deduped.append(enum)
        return deduped

    def _enumeration_signature(self, head: Optional[str], items: List[str], relation: str) -> Tuple[str, Tuple[str, ...], str]:
        """Canonical dedup signature for extracted enumerations."""
        head_norm = self._normalize_term(head or '')
        norm_items = tuple(self._normalize_term(i) for i in items if i)
        return head_norm, norm_items, relation

    def _score_list_boundary(self, sentence: str, items: List[str], relation: str) -> float:
        """Feature-weighted boundary classifier (lightweight ML-style scoring)."""
        if len(items) < 2:
            return 0.0

        marker_hits = sum(
            1 for m in (self.sequence_markers + self.example_markers + self.importance_markers + self.enumeration_lead_markers)
            if m in sentence
        )
        delimiter_hits = sentence.count(',') + sentence.count(';') + sentence.count(' සහ ') + sentence.count(' හා ') + sentence.count(' මෙන්ම ')
        numbered = 1.0 if self.numbering_pattern.search(sentence) else 0.0
        process_hit = 1.0 if any(m in sentence for m in self.process_markers) else 0.0
        argument_hit = 1.0 if any(m in sentence for m in self.argument_markers) else 0.0
        comparison_hit = 1.0 if any(m in sentence for m in self.comparison_markers) else 0.0

        item_count = min(1.0, len(items) / 5.0)
        delimiter_density = min(1.0, delimiter_hits / 4.0)
        marker_strength = min(1.0, marker_hits / 3.0)
        avg_len = sum(len(i) for i in items) / max(1, len(items))
        len_score = min(1.0, max(0.0, (avg_len - 3.0) / 10.0))

        relation_priors = {
            'sequence': 0.25,
            'example': 0.18,
            'requires': 0.22,
            'comparison': 0.30,
            'argument': 0.28,
            'group': 0.10,
        }
        prior = relation_priors.get(relation, 0.1)

        raw = (
            -1.15
            + (1.35 * item_count)
            + (0.70 * delimiter_density)
            + (0.80 * marker_strength)
            + (0.35 * len_score)
            + (0.40 * numbered)
            + (0.45 * process_hit)
            + (0.40 * argument_hit)
            + (0.45 * comparison_hit)
            + prior
        )
        return float(1.0 / (1.0 + math.exp(-raw)))

    def _extract_comparison_structure(self, sentence: str) -> Optional[Dict[str, Any]]:
        """Detect comparison/contrast structures, e.g., X vs Y or X සහ Y අතර වෙනස."""
        patterns = [
            r'([^,.;!?]{2,60}?)\s+(?:vs\.?|VS|වර්සස්|වඩා|ට වඩා|සසඳා|සසඳන විට)\s+([^,.;!?]{2,60})',
            r'([^,.;!?]{2,50}?)\s+සහ\s+([^,.;!?]{2,50}?)\s+අතර\s+(?:වෙනස|සමානතා|සසඳීම)',
        ]
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if not match:
                continue
            left = self.clean_label(self.numbering_pattern.sub(' ', match.group(1)).strip())
            right = self.clean_label(self.numbering_pattern.sub(' ', match.group(2)).strip())
            if left and right and left != right:
                score = self._score_list_boundary(sentence, [left, right], 'comparison')
                if score >= 0.5:
                    return {
                        'head': 'සසඳීම',
                        'items': [left, right],
                        'relation': 'comparison',
                        'confidence': round(score, 4),
                    }
        return None

    def _extract_argument_structure(self, sentence: str) -> Optional[Dict[str, Any]]:
        """Detect argument-like discourse lists common in student essays."""
        if not any(marker in sentence for marker in self.argument_markers):
            return None

        parts = [p.strip() for p in re.split(r'[;:]', sentence) if p.strip()]
        if len(parts) < 2:
            parts = [p.strip() for p in sentence.split('නමුත්') if p.strip()]

        items: List[str] = []
        for part in parts:
            cleaned = self.numbering_pattern.sub(' ', part).strip()
            tokens = [t for t in self._tokenize(cleaned) if t and t not in self.stop_words]
            if len(tokens) < 2:
                continue
            claim = ' '.join(tokens[:5])
            claim = self.clean_label(claim)
            if claim and claim not in items:
                items.append(claim)

        if len(items) < 2:
            return None

        score = self._score_list_boundary(sentence, items, 'argument')
        if score < 0.52:
            return None
        return {
            'head': 'තර්ක',
            'items': items,
            'relation': 'argument',
            'confidence': round(score, 4),
        }

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

        if not cleaned:
            return ''

        # Morphological normalization / lemmatization
        tokens = cleaned.split()
        normalized_tokens: List[str] = []
        for token in tokens:
            # citation form acts as a simple lemma
            lemma = self.morphology.get_citation_form(token)
            # fallback root normalization
            root = self.morphology.normalize_word(lemma or token)
            final = (root or lemma or token).strip()
            if final:
                normalized_tokens.append(final)

        return ' '.join(normalized_tokens)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode beyond NFC for Sinhala variant forms and hidden chars."""
        try:
            # Keep Sinhala canonical composition (avoid aggressive compatibility fold)
            norm = unicodedata.normalize('NFC', text)
            # Remove hidden chars except ZWJ/ZWNJ which are meaningful in Indic scripts
            norm = re.sub(r'[\u200B\uFEFF\u2060]', '', norm)
            # Canonicalize punctuation variants commonly seen in OCR/user input
            norm = (
                norm.replace('–', '-')
                .replace('—', '-')
                .replace('‘', "'")
                .replace('’', "'")
                .replace('“', '"')
                .replace('”', '"')
                .replace('…', '...')
            )
            # Normalize whitespace classes
            norm = re.sub(r'[\u00A0\u2000-\u200A\u202F\u205F\u3000]', ' ', norm)
            norm = re.sub(r'\s+', ' ', norm).strip()
            return unicodedata.normalize('NFC', norm)
        except Exception:
            return text

    def context_disambiguation_score(self, term: str, context: str) -> float:
        """
        Context-sensitive entity disambiguation score in [0,1].
        Higher means the term meaning aligns with surrounding sentence context.
        """
        if not term:
            return 0.5

        term_norm = self._normalize_term(term)
        context_norm = self._normalize_unicode(context or '')
        if not context_norm:
            return 0.5

        context_tokens = {self._normalize_term(t) for t in self._tokenize(context_norm)}
        context_tokens = {t for t in context_tokens if t and t not in self.stop_words}

        # Domain alignment score
        domain_best = 0.0
        domain_scores: Dict[str, float] = {}
        for _, anchors in self.disambiguation_domains.items():
            overlap = len(context_tokens & anchors)
            if overlap:
                score = min(1.0, overlap / max(2.0, len(anchors) * 0.2))
                domain_scores[_] = score
                domain_best = max(domain_best, score)

        # Ambiguous-term domain preference (same token can mean different things)
        ambiguous_bonus = 0.0
        term_root = self._normalize_term(self.morphology.extract_root(term_norm)) if term_norm else ''
        term_profile = self.ambiguous_term_domains.get(term_norm) or self.ambiguous_term_domains.get(term_root)
        if term_profile and domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            ambiguous_bonus = term_profile.get(best_domain, 0.0)

        # Local lexical coherence score (windowed overlap)
        term_parts = set(term_norm.split()) if term_norm else set()
        lexical = 0.0
        if term_parts and context_tokens:
            lexical = len(term_parts & context_tokens) / max(1, len(term_parts))

        # Optional semantic compatibility when embeddings are available
        semantic = 0.5
        if self.embeddings_model is not None:
            try:
                embs = self.embeddings_model.encode(
                    [term_norm, context_norm],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=16,
                )
                denom = (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])) + 1e-8
                semantic = float(np.dot(embs[0], embs[1]) / denom)
                semantic = max(0.0, min(1.0, semantic))
            except Exception:
                semantic = self._fallback_semantic_similarity(term_norm, context_norm)
        else:
            semantic = self._fallback_semantic_similarity(term_norm, context_norm)

        score = (0.20 * lexical) + (0.30 * domain_best) + (0.40 * semantic) + (0.10 * ambiguous_bonus)
        return float(max(0.0, min(1.0, score)))

    def _load_corpus_tuned_disambiguation_domains(
        self,
        base_domains: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """Tune domain lexicons with frequent terms mined from essay corpus CSVs."""
        domains = {k: set(v) for k, v in base_domains.items()}

        candidate_files = [
            Path(__file__).resolve().parent.parent / 'scoring-model-training' / 'akura_dataset.csv',
            Path(__file__).resolve().parent.parent / 'scoring-model-training' / 'sinhala_dataset_final_with_dyslexic.csv',
            Path(__file__).resolve().parent.parent / 'scoring-model-training' / 'sinhala_dataset_final.csv',
        ]

        domain_term_counts: Dict[str, Counter] = {d: Counter() for d in domains}
        total_rows = 0

        for csv_path in candidate_files:
            if not csv_path.exists():
                continue
            try:
                with open(csv_path, encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        total_rows += 1
                        topic = (row.get('essay_topic') or row.get('topic') or '').strip()
                        text = (row.get('essay_text') or row.get('input_text') or '').strip()
                        if not text:
                            continue

                        domain = self._infer_domain_from_topic(topic, text)
                        if domain not in domain_term_counts:
                            continue

                        tokens = [self._normalize_term(t) for t in re.findall(r'[\u0D80-\u0DFF]{2,}', self._normalize_unicode(text))]
                        for token in tokens:
                            if (
                                token
                                and token not in self.stop_words
                                and len(token) >= 3
                                and token not in self.postpositions
                                and token not in self.connectors
                            ):
                                lemma = self._normalize_term(self.morphology.get_citation_form(token))
                                domain_term_counts[domain][lemma or token] += 1
            except Exception as e:
                logger.warning("Failed to mine lexicon from %s: %s", csv_path, e)

        # Merge top high-frequency domain terms into base anchors
        for domain, counter in domain_term_counts.items():
            if not counter:
                continue
            for token, freq in counter.most_common(120):
                if freq >= 3:
                    domains[domain].add(token)

        logger.info(
            "Loaded corpus-tuned disambiguation lexicons from %d rows; sizes=%s",
            total_rows,
            {k: len(v) for k, v in domains.items()},
        )
        return domains

    def _infer_domain_from_topic(self, topic: str, text: str) -> str:
        """Infer domain label from essay topic (or early text if topic missing)."""
        candidate = self._normalize_unicode(topic or text[:120]).lower()
        rules = {
            'education': ['පාසල්', 'ශිෂ්‍ය', 'ගුරු', 'අධ්‍යාපන', 'ඉගෙන', 'විභාග', 'රචන'],
            'science': ['විද්‍ය', 'තාක්ෂණ', 'පර්යේෂණ', 'රසායන', 'භෞතික', 'ජීව'],
            'society': ['සමාජ', 'ජන', 'ප්‍රජා', 'සංස්කෘති', 'මාධ්‍ය', 'ගම්මාන'],
            'environment': ['පරිසර', 'වනය', 'ජල', 'ගංවතුර', 'දේශගුණ', 'දූෂණ'],
            'health': ['සෞඛ්‍ය', 'රෝග', 'ආහාර', 'ව්‍යායාම', 'වෛද්‍ය', 'නිරෝගී'],
        }
        best_domain = 'society'
        best_score = 0
        for domain, keys in rules.items():
            score = sum(1 for k in keys if k in candidate)
            if score > best_score:
                best_score = score
                best_domain = domain
        return best_domain

    def _build_ambiguous_term_domains(
        self,
        domains: Dict[str, Set[str]],
    ) -> Dict[str, Dict[str, float]]:
        """Build ambiguous-term domain preference map from overlapping domain lexicons."""
        term_to_domains: Dict[str, Set[str]] = defaultdict(set)
        for domain, tokens in domains.items():
            for t in tokens:
                term_to_domains[t].add(domain)

        # Keep terms present in 2+ domains as ambiguous and assign soft priors
        ambiguous: Dict[str, Dict[str, float]] = {}
        for term, dset in term_to_domains.items():
            if len(dset) < 2:
                continue
            # uniform prior over candidate domains
            prior = 1.0 / len(dset)
            ambiguous[term] = {d: prior for d in dset}

        # Inject known high-impact ambiguous Sinhala terms with stronger priors
        ambiguous.update({
            self._normalize_term('විද්‍යාව'): {'science': 0.7, 'education': 0.3},
            self._normalize_term('අධ්‍යයනය'): {'education': 0.55, 'science': 0.45},
            self._normalize_term('පද්ධතිය'): {'science': 0.5, 'society': 0.3, 'environment': 0.2},
            self._normalize_term('සංවර්ධනය'): {'society': 0.5, 'environment': 0.3, 'education': 0.2},
            self._normalize_term('රටාව'): {'science': 0.4, 'society': 0.4, 'education': 0.2},
            self._normalize_term('ජාලය'): {'science': 0.6, 'society': 0.4},
        })
        return ambiguous
    
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

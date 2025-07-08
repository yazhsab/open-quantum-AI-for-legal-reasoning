"""
Legal Text Preprocessor

This module handles preprocessing of legal documents and text for the
quantum-enhanced legal reasoning system. It includes specialized
preprocessing for Indian legal texts, case laws, and statutes.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

import re
import string
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import pandas as pd
import numpy as np
from loguru import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


@dataclass
class LegalEntity:
    """Represents a legal entity extracted from text."""
    text: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LegalCitation:
    """Represents a legal citation extracted from text."""
    text: str
    citation_type: str  # case, statute, article, section
    year: Optional[int] = None
    court: Optional[str] = None
    parties: Optional[List[str]] = None
    volume: Optional[str] = None
    page: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 1.0


@dataclass
class PreprocessedText:
    """Container for preprocessed legal text."""
    original_text: str
    cleaned_text: str
    tokens: List[str]
    sentences: List[str]
    entities: List[LegalEntity]
    citations: List[LegalCitation]
    legal_concepts: List[str]
    metadata: Dict[str, Any]
    
    # Linguistic features
    pos_tags: List[Tuple[str, str]]
    named_entities: List[Tuple[str, str]]
    
    # Legal-specific features
    case_references: List[str]
    statutory_references: List[str]
    legal_principles: List[str]
    
    # Preprocessing statistics
    preprocessing_stats: Dict[str, Any]


class LegalTextPreprocessor:
    """
    Comprehensive legal text preprocessor for Indian legal documents.
    
    This class handles various aspects of legal text preprocessing including:
    - Text cleaning and normalization
    - Legal entity recognition
    - Citation extraction and parsing
    - Legal concept identification
    - Multilingual support for Indian languages
    """
    
    def __init__(
        self,
        language: str = "en",
        enable_ner: bool = True,
        enable_citation_extraction: bool = True,
        enable_concept_extraction: bool = True,
        custom_legal_terms: Optional[List[str]] = None
    ):
        """
        Initialize the legal text preprocessor.
        
        Args:
            language: Primary language for processing
            enable_ner: Enable named entity recognition
            enable_citation_extraction: Enable legal citation extraction
            enable_concept_extraction: Enable legal concept extraction
            custom_legal_terms: Additional legal terms to recognize
        """
        self.language = language
        self.enable_ner = enable_ner
        self.enable_citation_extraction = enable_citation_extraction
        self.enable_concept_extraction = enable_concept_extraction
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Load legal vocabularies and patterns
        self._load_legal_vocabularies()
        self._load_citation_patterns()
        
        # Add custom legal terms
        if custom_legal_terms:
            self.legal_terms.update(custom_legal_terms)
        
        # Initialize preprocessing statistics
        self.stats = {
            "documents_processed": 0,
            "total_tokens": 0,
            "entities_extracted": 0,
            "citations_extracted": 0
        }
        
        logger.info(f"Legal text preprocessor initialized for language: {language}")
    
    def _initialize_nlp_components(self) -> None:
        """Initialize NLP components and models."""
        # Load spaCy model
        try:
            if self.language == "en":
                self.nlp = spacy.load("en_core_web_sm")
            else:
                # Fallback to basic English model
                self.nlp = spacy.load("en_core_web_sm")
                logger.warning(f"Using English model for language: {self.language}")
        except OSError:
            logger.error("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        
        # Add legal-specific stopwords
        legal_stopwords = {
            'whereas', 'whereof', 'wherein', 'whereby', 'herein', 'hereof',
            'thereof', 'therein', 'hereby', 'hereunder', 'thereunder',
            'aforesaid', 'aforementioned', 'said', 'such', 'same'
        }
        self.stop_words.update(legal_stopwords)
    
    def _load_legal_vocabularies(self) -> None:
        """Load legal vocabularies and term lists."""
        # Indian legal terms
        self.legal_terms = {
            # Courts
            'supreme court', 'high court', 'district court', 'sessions court',
            'magistrate court', 'family court', 'consumer court', 'tribunal',
            'appellate tribunal', 'civil court', 'criminal court',
            
            # Legal concepts
            'bail', 'anticipatory bail', 'interim bail', 'regular bail',
            'custody', 'remand', 'judicial custody', 'police custody',
            'cognizable', 'non-cognizable', 'bailable', 'non-bailable',
            'summons', 'warrant', 'arrest warrant', 'search warrant',
            'injunction', 'interim injunction', 'permanent injunction',
            'stay order', 'ex-parte', 'caveat', 'appeal', 'revision',
            'writ petition', 'habeas corpus', 'mandamus', 'certiorari',
            'prohibition', 'quo-warranto',
            
            # Property law
            'title deed', 'sale deed', 'gift deed', 'lease deed',
            'mortgage', 'easement', 'encumbrance', 'mutation',
            'partition', 'adverse possession', 'limitation',
            
            # Criminal law
            'first information report', 'fir', 'charge sheet', 'chargesheet',
            'investigation', 'trial', 'conviction', 'acquittal',
            'sentence', 'fine', 'imprisonment', 'probation',
            
            # Civil law
            'suit', 'plaint', 'written statement', 'issues',
            'evidence', 'examination', 'cross-examination', 'judgment',
            'decree', 'execution', 'attachment', 'garnishee',
            
            # Commercial law
            'negotiable instrument', 'cheque', 'promissory note',
            'bill of exchange', 'dishonour', 'bounce', 'default',
            'contract', 'agreement', 'consideration', 'breach'
        }
        
        # Indian legal acts and codes
        self.legal_acts = {
            'indian penal code', 'ipc', 'code of criminal procedure', 'crpc',
            'indian evidence act', 'constitution of india', 'civil procedure code',
            'cpc', 'negotiable instruments act', 'transfer of property act',
            'indian contract act', 'specific relief act', 'limitation act',
            'arbitration and conciliation act', 'consumer protection act',
            'information technology act', 'companies act', 'income tax act',
            'goods and services tax act', 'gst act', 'motor vehicle act',
            'prevention of corruption act', 'dowry prohibition act',
            'domestic violence act', 'juvenile justice act'
        }
        
        # Legal positions and roles
        self.legal_roles = {
            'judge', 'justice', 'chief justice', 'magistrate', 'advocate',
            'senior advocate', 'counsel', 'prosecutor', 'public prosecutor',
            'additional public prosecutor', 'government pleader',
            'standing counsel', 'amicus curiae', 'plaintiff', 'defendant',
            'petitioner', 'respondent', 'appellant', 'appellee',
            'complainant', 'accused', 'witness', 'expert witness'
        }
    
    def _load_citation_patterns(self) -> None:
        """Load regex patterns for legal citation extraction."""
        self.citation_patterns = {
            # Indian Supreme Court citations
            'supreme_court': [
                r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',  # (2020) 5 SCC 123
                r'\d{4}\s*\(\d+\)\s*SCC\s*\d+',  # 2020 (5) SCC 123
                r'AIR\s*\d{4}\s*SC\s*\d+',       # AIR 2020 SC 123
                r'\d{4}\s*\d+\s*SCR\s*\d+',      # 2020 5 SCR 123
            ],
            
            # High Court citations
            'high_court': [
                r'AIR\s*\d{4}\s*[A-Z]+\s*\d+',   # AIR 2020 Del 123
                r'\d{4}\s*\(\d+\)\s*[A-Z]+LJ\s*\d+',  # 2020 (5) DLT 123
                r'\(\d{4}\)\s*\d+\s*[A-Z]+\s*\d+',    # (2020) 5 Bom 123
            ],
            
            # Constitutional articles
            'constitution': [
                r'Article\s*\d+(?:\(\d+\))?(?:\([a-z]\))?',  # Article 21, Article 14(1)(a)
                r'Art\.?\s*\d+(?:\(\d+\))?(?:\([a-z]\))?',   # Art. 21, Art 14(1)(a)
            ],
            
            # Statutory sections
            'statute': [
                r'Section\s*\d+(?:[A-Z])?(?:\(\d+\))?(?:\([a-z]\))?',  # Section 302, Section 498A(1)(a)
                r'Sec\.?\s*\d+(?:[A-Z])?(?:\(\d+\))?(?:\([a-z]\))?',   # Sec. 302, Sec 498A(1)(a)
                r'S\.?\s*\d+(?:[A-Z])?(?:\(\d+\))?(?:\([a-z]\))?',     # S. 302, S 498A(1)(a)
            ],
            
            # Case names
            'case_name': [
                r'[A-Z][a-zA-Z\s]+\s+v\.?\s+[A-Z][a-zA-Z\s]+',  # Plaintiff v. Defendant
                r'[A-Z][a-zA-Z\s]+\s+vs\.?\s+[A-Z][a-zA-Z\s]+', # Plaintiff vs. Defendant
            ]
        }
    
    async def preprocess_text(
        self,
        text: str,
        document_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> PreprocessedText:
        """
        Preprocess legal text asynchronously.
        
        Args:
            text: Input legal text
            document_type: Type of legal document
            metadata: Additional metadata about the document
            
        Returns:
            PreprocessedText object with all extracted information
        """
        if metadata is None:
            metadata = {}
        
        logger.debug(f"Preprocessing text of length {len(text)}")
        
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        # Tokenization
        tokens = self._tokenize(cleaned_text)
        sentences = self._sentence_tokenize(cleaned_text)
        
        # Parallel processing of different extraction tasks
        tasks = []
        
        if self.enable_ner:
            tasks.append(self._extract_entities(cleaned_text))
        else:
            tasks.append(asyncio.create_task(self._empty_list()))
        
        if self.enable_citation_extraction:
            tasks.append(self._extract_citations(cleaned_text))
        else:
            tasks.append(asyncio.create_task(self._empty_list()))
        
        if self.enable_concept_extraction:
            tasks.append(self._extract_legal_concepts(cleaned_text))
        else:
            tasks.append(asyncio.create_task(self._empty_list()))
        
        # Additional linguistic analysis
        tasks.extend([
            self._pos_tagging(tokens),
            self._named_entity_recognition(cleaned_text),
            self._extract_case_references(cleaned_text),
            self._extract_statutory_references(cleaned_text),
            self._extract_legal_principles(cleaned_text)
        ])
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        entities, citations, legal_concepts, pos_tags, named_entities, \
        case_references, statutory_references, legal_principles = results
        
        # Calculate preprocessing statistics
        preprocessing_stats = {
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "num_tokens": len(tokens),
            "num_sentences": len(sentences),
            "num_entities": len(entities),
            "num_citations": len(citations),
            "num_legal_concepts": len(legal_concepts),
            "document_type": document_type
        }
        
        # Update global statistics
        self.stats["documents_processed"] += 1
        self.stats["total_tokens"] += len(tokens)
        self.stats["entities_extracted"] += len(entities)
        self.stats["citations_extracted"] += len(citations)
        
        # Create preprocessed text object
        preprocessed = PreprocessedText(
            original_text=text,
            cleaned_text=cleaned_text,
            tokens=tokens,
            sentences=sentences,
            entities=entities,
            citations=citations,
            legal_concepts=legal_concepts,
            metadata=metadata,
            pos_tags=pos_tags,
            named_entities=named_entities,
            case_references=case_references,
            statutory_references=statutory_references,
            legal_principles=legal_principles,
            preprocessing_stats=preprocessing_stats
        )
        
        logger.debug(f"Preprocessing completed: {preprocessing_stats}")
        
        return preprocessed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize legal text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s*\d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Clean up legal document formatting
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines
        text = re.sub(r'^\s+|\s+$', '', text)  # Leading/trailing whitespace
        
        # Normalize legal abbreviations
        abbreviation_map = {
            r'\bv\.\s+': 'v. ',
            r'\bvs\.\s+': 'vs. ',
            r'\bJ\.\s+': 'J. ',
            r'\bCJ\.\s+': 'CJ. ',
            r'\bSec\.\s+': 'Section ',
            r'\bArt\.\s+': 'Article ',
            r'\bPara\.\s+': 'Para ',
            r'\bCl\.\s+': 'Clause ',
        }
        
        for pattern, replacement in abbreviation_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Use NLTK word tokenizer
        tokens = word_tokenize(text)
        
        # Filter out punctuation and empty tokens
        tokens = [
            token.lower() for token in tokens
            if token not in string.punctuation and token.strip()
        ]
        
        # Remove stopwords if configured
        if hasattr(self, 'remove_stopwords') and self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def _sentence_tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        sentences = sent_tokenize(text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    async def _extract_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal entities from text."""
        entities = []
        
        if self.nlp is None:
            return entities
        
        # Use spaCy for NER
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # Map spaCy entity types to legal entity types
            entity_type = self._map_entity_type(ent.label_)
            
            legal_entity = LegalEntity(
                text=ent.text,
                entity_type=entity_type,
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores
                metadata={"spacy_label": ent.label_}
            )
            
            entities.append(legal_entity)
        
        # Extract legal-specific entities
        legal_entities = await self._extract_legal_specific_entities(text)
        entities.extend(legal_entities)
        
        return entities
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """Map spaCy entity labels to legal entity types."""
        mapping = {
            "PERSON": "person",
            "ORG": "organization",
            "GPE": "location",
            "DATE": "date",
            "MONEY": "monetary_amount",
            "LAW": "legal_reference",
            "CARDINAL": "number",
            "ORDINAL": "ordinal"
        }
        
        return mapping.get(spacy_label, "other")
    
    async def _extract_legal_specific_entities(self, text: str) -> List[LegalEntity]:
        """Extract legal-specific entities."""
        entities = []
        
        # Extract court names
        court_patterns = [
            r'Supreme Court of India',
            r'High Court of [A-Za-z\s]+',
            r'[A-Za-z\s]+ High Court',
            r'District Court of [A-Za-z\s]+',
            r'[A-Za-z\s]+ District Court',
            r'Sessions Court',
            r'Magistrate Court'
        ]
        
        for pattern in court_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = LegalEntity(
                    text=match.group(),
                    entity_type="court",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                )
                entities.append(entity)
        
        # Extract legal acts
        for act in self.legal_acts:
            pattern = re.escape(act)
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = LegalEntity(
                    text=match.group(),
                    entity_type="legal_act",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8
                )
                entities.append(entity)
        
        return entities
    
    async def _extract_citations(self, text: str) -> List[LegalCitation]:
        """Extract legal citations from text."""
        citations = []
        
        for citation_type, patterns in self.citation_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    citation = LegalCitation(
                        text=match.group(),
                        citation_type=citation_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9
                    )
                    
                    # Parse additional information from citation
                    citation = self._parse_citation_details(citation)
                    citations.append(citation)
        
        return citations
    
    def _parse_citation_details(self, citation: LegalCitation) -> LegalCitation:
        """Parse additional details from citation text."""
        text = citation.text
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            citation.year = int(year_match.group())
        
        # Extract court information for case citations
        if citation.citation_type in ['supreme_court', 'high_court']:
            if 'SC' in text:
                citation.court = 'Supreme Court'
            elif any(hc in text for hc in ['Del', 'Bom', 'Cal', 'Mad', 'All']):
                citation.court = 'High Court'
        
        return citation
    
    async def _extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts from text."""
        concepts = []
        
        # Extract concepts based on legal terms
        text_lower = text.lower()
        
        for term in self.legal_terms:
            if term in text_lower:
                concepts.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    async def _pos_tagging(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Perform part-of-speech tagging."""
        return pos_tag(tokens)
    
    async def _named_entity_recognition(self, text: str) -> List[Tuple[str, str]]:
        """Perform named entity recognition using NLTK."""
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        chunks = ne_chunk(pos_tags)
        
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_text = ' '.join([token for token, pos in chunk.leaves()])
                entities.append((entity_text, chunk.label()))
        
        return entities
    
    async def _extract_case_references(self, text: str) -> List[str]:
        """Extract case references from text."""
        case_patterns = [
            r'[A-Z][a-zA-Z\s]+\s+v\.?\s+[A-Z][a-zA-Z\s]+',
            r'[A-Z][a-zA-Z\s]+\s+vs\.?\s+[A-Z][a-zA-Z\s]+',
            r'In re:?\s+[A-Z][a-zA-Z\s]+',
            r'Ex parte:?\s+[A-Z][a-zA-Z\s]+'
        ]
        
        references = []
        for pattern in case_patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    async def _extract_statutory_references(self, text: str) -> List[str]:
        """Extract statutory references from text."""
        statutory_patterns = [
            r'Section\s+\d+(?:[A-Z])?(?:\(\d+\))?(?:\([a-z]\))?',
            r'Article\s+\d+(?:\(\d+\))?(?:\([a-z]\))?',
            r'Rule\s+\d+(?:\(\d+\))?(?:\([a-z]\))?',
            r'Order\s+\d+(?:\(\d+\))?(?:\([a-z]\))?',
            r'Schedule\s+[IVX]+',
            r'Chapter\s+[IVX]+',
            r'Part\s+[IVX]+'
        ]
        
        references = []
        for pattern in statutory_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    async def _extract_legal_principles(self, text: str) -> List[str]:
        """Extract legal principles and maxims from text."""
        # Common legal principles and maxims
        principles = [
            'innocent until proven guilty',
            'burden of proof',
            'beyond reasonable doubt',
            'preponderance of evidence',
            'due process',
            'natural justice',
            'audi alteram partem',
            'nemo judex in causa sua',
            'res judicata',
            'double jeopardy',
            'habeas corpus',
            'certiorari',
            'mandamus',
            'prohibition',
            'quo warranto'
        ]
        
        found_principles = []
        text_lower = text.lower()
        
        for principle in principles:
            if principle in text_lower:
                found_principles.append(principle)
        
        return found_principles
    
    async def _empty_list(self) -> List:
        """Return empty list for disabled features."""
        return []
    
    def preprocess_batch(
        self,
        texts: List[str],
        document_types: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = 4
    ) -> List[PreprocessedText]:
        """
        Preprocess multiple texts in parallel.
        
        Args:
            texts: List of input texts
            document_types: List of document types
            metadata_list: List of metadata dictionaries
            max_workers: Maximum number of worker threads
            
        Returns:
            List of PreprocessedText objects
        """
        if document_types is None:
            document_types = ["general"] * len(texts)
        
        if metadata_list is None:
            metadata_list = [{}] * len(texts)
        
        async def process_all():
            tasks = []
            for i, text in enumerate(texts):
                task = self.preprocess_text(
                    text,
                    document_types[i],
                    metadata_list[i]
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(process_all())
        finally:
            loop.close()
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset preprocessing statistics."""
        self.stats = {
            "documents_processed": 0,
            "total_tokens": 0,
            "entities_extracted": 0,
            "citations_extracted": 0
        }
    
    def add_legal_terms(self, terms: List[str]) -> None:
        """Add custom legal terms to the vocabulary."""
        self.legal_terms.update(terms)
        logger.info(f"Added {len(terms)} custom legal terms")
    
    def add_citation_pattern(self, pattern: str, citation_type: str) -> None:
        """Add custom citation pattern."""
        if citation_type not in self.citation_patterns:
            self.citation_patterns[citation_type] = []
        
        self.citation_patterns[citation_type].append(pattern)
        logger.info(f"Added citation pattern for type: {citation_type}")
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save legal vocabulary to file."""
        vocabulary = {
            'legal_terms': list(self.legal_terms),
            'legal_acts': list(self.legal_acts),
            'legal_roles': list(self.legal_roles),
            'citation_patterns': self.citation_patterns
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(vocabulary, f, indent=2)
        
        logger.info(f"Legal vocabulary saved to: {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """Load legal vocabulary from file."""
        import json
        
        try:
            with open(filepath, 'r') as f:
                vocabulary = json.load(f)
            
            if 'legal_terms' in vocabulary:
                self.legal_terms.update(vocabulary['legal_terms'])
            
            if 'legal_acts' in vocabulary:
                self.legal_acts.update(vocabulary['legal_acts'])
            
            if 'legal_roles' in vocabulary:
                self.legal_roles.update(vocabulary['legal_roles'])
            
            if 'citation_patterns' in vocabulary:
                for citation_type, patterns in vocabulary['citation_patterns'].items():
                    if citation_type not in self.citation_patterns:
                        self.citation_patterns[citation_type] = []
                    self.citation_patterns[citation_type].extend(patterns)
            
            logger.info(f"Legal vocabulary loaded from: {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading vocabulary from {filepath}: {e}")
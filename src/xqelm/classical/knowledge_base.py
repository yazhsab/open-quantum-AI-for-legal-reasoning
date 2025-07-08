"""
Legal Knowledge Base

This module manages the legal knowledge base including case laws,
statutes, legal principles, and domain-specific knowledge for
the quantum-enhanced legal reasoning system.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, date
import asyncio
import hashlib
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger
import sqlite3
import aiosqlite
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import faiss
from sentence_transformers import SentenceTransformer


class DocumentType(Enum):
    """Types of legal documents."""
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    CONSTITUTIONAL_PROVISION = "constitutional_provision"
    LEGAL_PRINCIPLE = "legal_principle"
    LEGAL_MAXIM = "legal_maxim"
    COMMENTARY = "commentary"
    JOURNAL_ARTICLE = "journal_article"


class JurisdictionLevel(Enum):
    """Levels of legal jurisdiction."""
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"
    DISTRICT_COURT = "district_court"
    TRIBUNAL = "tribunal"
    CONSTITUTIONAL = "constitutional"
    STATUTORY = "statutory"


@dataclass
class LegalDocument:
    """Represents a legal document in the knowledge base."""
    id: str
    title: str
    content: str
    document_type: DocumentType
    jurisdiction: str
    court: Optional[str] = None
    date_decided: Optional[date] = None
    citation: Optional[str] = None
    judges: Optional[List[str]] = None
    parties: Optional[List[str]] = None
    legal_concepts: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    summary: Optional[str] = None
    holding: Optional[str] = None
    ratio: Optional[str] = None
    obiter_dicta: Optional[str] = None
    precedent_value: Optional[float] = None
    relevance_score: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LegalPrinciple:
    """Represents a legal principle or rule."""
    id: str
    name: str
    description: str
    source_documents: List[str]
    jurisdiction: str
    legal_area: str
    confidence: float
    applications: List[str]
    exceptions: List[str]
    related_principles: List[str]
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LegalConcept:
    """Represents a legal concept."""
    id: str
    name: str
    definition: str
    synonyms: List[str]
    related_concepts: List[str]
    legal_area: str
    importance_score: float
    usage_frequency: int
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """Represents a search result from the knowledge base."""
    document: LegalDocument
    similarity_score: float
    relevance_score: float
    match_type: str  # exact, semantic, conceptual
    matched_terms: List[str]
    explanation: Optional[str] = None


# SQLAlchemy models
Base = declarative_base()


class DocumentModel(Base):
    """SQLAlchemy model for legal documents."""
    __tablename__ = 'legal_documents'
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    document_type = Column(String, nullable=False)
    jurisdiction = Column(String, nullable=False)
    court = Column(String)
    date_decided = Column(DateTime)
    citation = Column(String)
    judges = Column(Text)  # JSON string
    parties = Column(Text)  # JSON string
    legal_concepts = Column(Text)  # JSON string
    keywords = Column(Text)  # JSON string
    summary = Column(Text)
    holding = Column(Text)
    ratio = Column(Text)
    obiter_dicta = Column(Text)
    precedent_value = Column(Float)
    relevance_score = Column(Float)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class LegalKnowledgeBase:
    """
    Comprehensive legal knowledge base for quantum-enhanced legal reasoning.
    
    This class manages storage, retrieval, and semantic search of legal
    documents, principles, and concepts using both traditional and
    quantum-enhanced methods.
    """
    
    def __init__(
        self,
        db_path: str = "legal_knowledge.db",
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_index_path: str = "legal_vectors.index",
        jurisdiction: str = "india"
    ):
        """
        Initialize the legal knowledge base.
        
        Args:
            db_path: Path to SQLite database
            embedding_model: Sentence transformer model for embeddings
            vector_index_path: Path to FAISS vector index
            jurisdiction: Primary jurisdiction
        """
        self.db_path = db_path
        self.vector_index_path = vector_index_path
        self.jurisdiction = jurisdiction
        
        # Initialize database
        self._initialize_database()
        
        # Initialize embedding model
        self._initialize_embedding_model(embedding_model)
        
        # Initialize vector index
        self._initialize_vector_index()
        
        # Load legal taxonomies
        self._load_legal_taxonomies()
        
        # Initialize statistics
        self.stats = {
            "total_documents": 0,
            "total_principles": 0,
            "total_concepts": 0,
            "search_queries": 0,
            "cache_hits": 0
        }
        
        # Initialize cache
        self.search_cache = {}
        self.max_cache_size = 1000
        
        logger.info(f"Legal knowledge base initialized for {jurisdiction}")
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database."""
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create async connection
        self.db_path_async = self.db_path
    
    def _initialize_embedding_model(self, model_name: str) -> None:
        """Initialize sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            # Fallback to a basic model
            self.embedding_model = None
            self.embedding_dim = 384
    
    def _initialize_vector_index(self) -> None:
        """Initialize FAISS vector index."""
        try:
            if Path(self.vector_index_path).exists():
                self.vector_index = faiss.read_index(self.vector_index_path)
                logger.info(f"Loaded existing vector index: {self.vector_index_path}")
            else:
                # Create new index
                self.vector_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
                logger.info(f"Created new vector index with dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize vector index: {e}")
            self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Document ID mapping
        self.doc_id_to_index = {}
        self.index_to_doc_id = {}
    
    def _load_legal_taxonomies(self) -> None:
        """Load legal taxonomies and classification systems."""
        # Indian legal subject classification
        self.legal_subjects = {
            "constitutional_law": [
                "fundamental_rights", "directive_principles", "federal_structure",
                "judicial_review", "emergency_provisions"
            ],
            "criminal_law": [
                "offenses", "procedure", "evidence", "bail", "sentencing",
                "investigation", "trial", "appeals"
            ],
            "civil_law": [
                "contracts", "torts", "property", "family", "succession",
                "civil_procedure", "limitation", "specific_relief"
            ],
            "commercial_law": [
                "company_law", "banking", "insurance", "securities",
                "intellectual_property", "competition", "arbitration"
            ],
            "administrative_law": [
                "government_contracts", "public_services", "regulatory",
                "judicial_review", "natural_justice"
            ],
            "tax_law": [
                "income_tax", "goods_services_tax", "customs", "excise",
                "tax_procedure", "tax_appeals"
            ]
        }
        
        # Court hierarchy
        self.court_hierarchy = {
            "supreme_court": 5,
            "high_court": 4,
            "district_court": 3,
            "sessions_court": 2,
            "magistrate_court": 1
        }
        
        # Legal citation patterns
        self.citation_patterns = {
            "supreme_court": [
                r'\(\d{4}\)\s*\d+\s*SCC\s*\d+',
                r'AIR\s*\d{4}\s*SC\s*\d+'
            ],
            "high_court": [
                r'AIR\s*\d{4}\s*[A-Z]+\s*\d+',
                r'\d{4}\s*\(\d+\)\s*[A-Z]+LJ\s*\d+'
            ]
        }
    
    async def add_document(self, document: LegalDocument) -> bool:
        """
        Add a legal document to the knowledge base.
        
        Args:
            document: Legal document to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding if not provided
            if document.embedding is None and self.embedding_model:
                text_to_embed = f"{document.title} {document.content}"
                document.embedding = self.embedding_model.encode(text_to_embed)
            
            # Add to database
            async with aiosqlite.connect(self.db_path_async) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO legal_documents 
                    (id, title, content, document_type, jurisdiction, court, 
                     date_decided, citation, judges, parties, legal_concepts, 
                     keywords, summary, holding, ratio, obiter_dicta, 
                     precedent_value, relevance_score, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.id, document.title, document.content,
                    document.document_type.value, document.jurisdiction,
                    document.court, document.date_decided, document.citation,
                    json.dumps(document.judges) if document.judges else None,
                    json.dumps(document.parties) if document.parties else None,
                    json.dumps(document.legal_concepts) if document.legal_concepts else None,
                    json.dumps(document.keywords) if document.keywords else None,
                    document.summary, document.holding, document.ratio,
                    document.obiter_dicta, document.precedent_value,
                    document.relevance_score,
                    json.dumps(document.metadata) if document.metadata else None,
                    document.created_at, document.updated_at
                ))
                await db.commit()
            
            # Add to vector index
            if document.embedding is not None:
                self._add_to_vector_index(document.id, document.embedding)
            
            # Update statistics
            self.stats["total_documents"] += 1
            
            logger.debug(f"Added document: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            return False
    
    def _add_to_vector_index(self, doc_id: str, embedding: np.ndarray) -> None:
        """Add document embedding to vector index."""
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add to index
        index_id = self.vector_index.ntotal
        self.vector_index.add(embedding.reshape(1, -1))
        
        # Update mappings
        self.doc_id_to_index[doc_id] = index_id
        self.index_to_doc_id[index_id] = doc_id
    
    async def search_documents(
        self,
        query: str,
        document_types: Optional[List[DocumentType]] = None,
        jurisdiction: Optional[str] = None,
        date_range: Optional[Tuple[date, date]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Search for legal documents using semantic similarity.
        
        Args:
            query: Search query
            document_types: Filter by document types
            jurisdiction: Filter by jurisdiction
            date_range: Filter by date range
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        # Check cache
        cache_key = self._generate_cache_key(
            query, document_types, jurisdiction, date_range, limit
        )
        
        if cache_key in self.search_cache:
            self.stats["cache_hits"] += 1
            return self.search_cache[cache_key]
        
        self.stats["search_queries"] += 1
        
        try:
            # Generate query embedding
            if self.embedding_model:
                query_embedding = self.embedding_model.encode(query)
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
            else:
                # Fallback to keyword search
                return await self._keyword_search(
                    query, document_types, jurisdiction, date_range, limit
                )
            
            # Vector search
            scores, indices = self.vector_index.search(
                query_embedding.reshape(1, -1), min(limit * 2, self.vector_index.ntotal)
            )
            
            # Get document IDs
            candidate_doc_ids = []
            candidate_scores = []
            
            for i, (score, index) in enumerate(zip(scores[0], indices[0])):
                if index in self.index_to_doc_id and score >= similarity_threshold:
                    doc_id = self.index_to_doc_id[index]
                    candidate_doc_ids.append(doc_id)
                    candidate_scores.append(float(score))
            
            # Retrieve documents from database
            documents = await self._get_documents_by_ids(candidate_doc_ids)
            
            # Apply filters
            filtered_results = []
            for doc, score in zip(documents, candidate_scores):
                if self._apply_filters(doc, document_types, jurisdiction, date_range):
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(doc, query, score)
                    
                    # Find matched terms
                    matched_terms = self._find_matched_terms(doc, query)
                    
                    result = SearchResult(
                        document=doc,
                        similarity_score=score,
                        relevance_score=relevance_score,
                        match_type="semantic",
                        matched_terms=matched_terms,
                        explanation=f"Semantic similarity: {score:.3f}"
                    )
                    filtered_results.append(result)
            
            # Sort by relevance score
            filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit results
            results = filtered_results[:limit]
            
            # Cache results
            if len(self.search_cache) < self.max_cache_size:
                self.search_cache[cache_key] = results
            
            logger.debug(f"Search returned {len(results)} results for query: {query[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        document_types: Optional[List[DocumentType]],
        jurisdiction: Optional[str],
        date_range: Optional[Tuple[date, date]],
        limit: int
    ) -> List[SearchResult]:
        """Fallback keyword search."""
        # Simple keyword matching
        keywords = query.lower().split()
        
        # Build SQL query
        sql_conditions = []
        params = []
        
        # Text search
        text_conditions = []
        for keyword in keywords:
            text_conditions.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        
        if text_conditions:
            sql_conditions.append(f"({' OR '.join(text_conditions)})")
        
        # Apply filters
        if document_types:
            type_conditions = " OR ".join(["document_type = ?"] * len(document_types))
            sql_conditions.append(f"({type_conditions})")
            params.extend([dt.value for dt in document_types])
        
        if jurisdiction:
            sql_conditions.append("jurisdiction = ?")
            params.append(jurisdiction)
        
        if date_range:
            sql_conditions.append("date_decided BETWEEN ? AND ?")
            params.extend(date_range)
        
        # Build final query
        where_clause = " AND ".join(sql_conditions) if sql_conditions else "1=1"
        sql_query = f"""
            SELECT * FROM legal_documents 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        
        # Execute query
        async with aiosqlite.connect(self.db_path_async) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql_query, params) as cursor:
                rows = await cursor.fetchall()
        
        # Convert to SearchResult objects
        results = []
        for row in rows:
            doc = self._row_to_document(row)
            
            # Calculate simple relevance score
            relevance_score = self._calculate_keyword_relevance(doc, keywords)
            matched_terms = [kw for kw in keywords if kw in doc.content.lower()]
            
            result = SearchResult(
                document=doc,
                similarity_score=relevance_score,
                relevance_score=relevance_score,
                match_type="keyword",
                matched_terms=matched_terms,
                explanation=f"Keyword match: {len(matched_terms)}/{len(keywords)} terms"
            )
            results.append(result)
        
        return results
    
    async def _get_documents_by_ids(self, doc_ids: List[str]) -> List[LegalDocument]:
        """Retrieve documents by IDs."""
        if not doc_ids:
            return []
        
        placeholders = ",".join(["?"] * len(doc_ids))
        sql_query = f"SELECT * FROM legal_documents WHERE id IN ({placeholders})"
        
        async with aiosqlite.connect(self.db_path_async) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql_query, doc_ids) as cursor:
                rows = await cursor.fetchall()
        
        return [self._row_to_document(row) for row in rows]
    
    def _row_to_document(self, row: aiosqlite.Row) -> LegalDocument:
        """Convert database row to LegalDocument."""
        return LegalDocument(
            id=row['id'],
            title=row['title'],
            content=row['content'],
            document_type=DocumentType(row['document_type']),
            jurisdiction=row['jurisdiction'],
            court=row['court'],
            date_decided=datetime.fromisoformat(row['date_decided']).date() if row['date_decided'] else None,
            citation=row['citation'],
            judges=json.loads(row['judges']) if row['judges'] else None,
            parties=json.loads(row['parties']) if row['parties'] else None,
            legal_concepts=json.loads(row['legal_concepts']) if row['legal_concepts'] else None,
            keywords=json.loads(row['keywords']) if row['keywords'] else None,
            summary=row['summary'],
            holding=row['holding'],
            ratio=row['ratio'],
            obiter_dicta=row['obiter_dicta'],
            precedent_value=row['precedent_value'],
            relevance_score=row['relevance_score'],
            metadata=json.loads(row['metadata']) if row['metadata'] else None,
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )
    
    def _apply_filters(
        self,
        document: LegalDocument,
        document_types: Optional[List[DocumentType]],
        jurisdiction: Optional[str],
        date_range: Optional[Tuple[date, date]]
    ) -> bool:
        """Apply search filters to document."""
        if document_types and document.document_type not in document_types:
            return False
        
        if jurisdiction and document.jurisdiction != jurisdiction:
            return False
        
        if date_range and document.date_decided:
            start_date, end_date = date_range
            if not (start_date <= document.date_decided <= end_date):
                return False
        
        return True
    
    def _calculate_relevance_score(
        self,
        document: LegalDocument,
        query: str,
        similarity_score: float
    ) -> float:
        """Calculate relevance score combining multiple factors."""
        # Base similarity score
        relevance = similarity_score
        
        # Boost for document type importance
        type_boost = {
            DocumentType.CASE_LAW: 1.2,
            DocumentType.STATUTE: 1.1,
            DocumentType.CONSTITUTIONAL_PROVISION: 1.3,
            DocumentType.LEGAL_PRINCIPLE: 1.0
        }
        relevance *= type_boost.get(document.document_type, 1.0)
        
        # Boost for court hierarchy
        if document.court:
            court_boost = self.court_hierarchy.get(document.court.lower(), 1.0) / 5.0
            relevance *= (1.0 + court_boost * 0.2)
        
        # Boost for precedent value
        if document.precedent_value:
            relevance *= (1.0 + document.precedent_value * 0.1)
        
        # Recency boost (newer documents get slight boost)
        if document.date_decided:
            years_old = (date.today() - document.date_decided).days / 365.25
            recency_boost = max(0.0, 1.0 - years_old * 0.01)  # 1% decay per year
            relevance *= (1.0 + recency_boost * 0.1)
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    def _find_matched_terms(self, document: LegalDocument, query: str) -> List[str]:
        """Find terms from query that match in document."""
        query_terms = set(query.lower().split())
        doc_text = f"{document.title} {document.content}".lower()
        
        matched = []
        for term in query_terms:
            if term in doc_text:
                matched.append(term)
        
        return matched
    
    def _calculate_keyword_relevance(self, document: LegalDocument, keywords: List[str]) -> float:
        """Calculate relevance score for keyword search."""
        doc_text = f"{document.title} {document.content}".lower()
        
        matches = 0
        for keyword in keywords:
            if keyword in doc_text:
                matches += 1
        
        return matches / len(keywords) if keywords else 0.0
    
    def _generate_cache_key(
        self,
        query: str,
        document_types: Optional[List[DocumentType]],
        jurisdiction: Optional[str],
        date_range: Optional[Tuple[date, date]],
        limit: int
    ) -> str:
        """Generate cache key for search parameters."""
        key_parts = [
            query,
            str(sorted([dt.value for dt in document_types]) if document_types else []),
            jurisdiction or "",
            str(date_range) if date_range else "",
            str(limit)
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_document_by_id(self, doc_id: str) -> Optional[LegalDocument]:
        """Get document by ID."""
        async with aiosqlite.connect(self.db_path_async) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM legal_documents WHERE id = ?", (doc_id,)
            ) as cursor:
                row = await cursor.fetchone()
        
        return self._row_to_document(row) if row else None
    
    async def update_document(self, document: LegalDocument) -> bool:
        """Update existing document."""
        document.updated_at = datetime.now()
        return await self.add_document(document)  # Uses INSERT OR REPLACE
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from knowledge base."""
        try:
            async with aiosqlite.connect(self.db_path_async) as db:
                await db.execute("DELETE FROM legal_documents WHERE id = ?", (doc_id,))
                await db.commit()
            
            # Remove from vector index (would require rebuilding index)
            if doc_id in self.doc_id_to_index:
                # Mark for removal - actual removal requires index rebuild
                del self.doc_id_to_index[doc_id]
            
            self.stats["total_documents"] -= 1
            logger.debug(f"Deleted document: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    async def get_similar_documents(
        self,
        document_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Find documents similar to a given document."""
        # Get the reference document
        ref_doc = await self.get_document_by_id(document_id)
        if not ref_doc:
            return []
        
        # Use document content as query
        query = f"{ref_doc.title} {ref_doc.summary or ref_doc.content[:500]}"
        
        # Search for similar documents
        results = await self.search_documents(
            query=query,
            limit=limit + 1,  # +1 to exclude self
            similarity_threshold=similarity_threshold
        )
        
        # Filter out the reference document itself
        similar_docs = [r for r in results if r.document.id != document_id]
        
        return similar_docs[:limit]
    
    def save_vector_index(self) -> None:
        """Save vector index to disk."""
        try:
            faiss.write_index(self.vector_index, self.vector_index_path)
            
            # Save mappings
            mappings = {
                'doc_id_to_index': self.doc_id_to_index,
                'index_to_doc_id': self.index_to_doc_id
            }
            
            mapping_path = self.vector_index_path.replace('.index', '_mappings.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(mappings, f)
            
            logger.info(f"Vector index saved to {self.vector_index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
    
    def load_vector_index(self) -> None:
        """Load vector index from disk."""
        try:
            if Path(self.vector_index_path).exists():
                self.vector_index = faiss.read_index(self.vector_index_path)
                
                # Load mappings
                mapping_path = self.vector_index_path.replace('.index', '_mappings.pkl')
                if Path(mapping_path).exists():
                    with open(mapping_path, 'rb') as f:
                        mappings = pickle.load(f)
                    
                    self.doc_id_to_index = mappings['doc_id_to_index']
                    self.index_to_doc_id = mappings['index_to_doc_id']
                
                logger.info(f"Vector index loaded from {self.vector_index_path}")
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
    
    async def rebuild_vector_index(self) -> None:
        """Rebuild vector index from all documents."""
        logger.info("Rebuilding vector index...")
        
        # Create new index
        self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
        self.doc_id_to_index = {}
        self.index_to_doc_id = {}
        
        # Get all documents
        async with aiosqlite.connect(self.db_path_async) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT id, title, content FROM legal_documents") as cursor:
                async for row in cursor:
                    if self.embedding_model:
                        text = f"{row['title']} {row['content']}"
                        embedding = self.embedding_model.encode(text)
                        self._add_to_vector_index(row['id'], embedding)
        
        # Save the rebuilt index
        self.save_vector_index()
        
        logger.info(f"Vector index rebuilt with {self.vector_index.ntotal} documents")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            **self.stats,
            "vector_index_size": self.vector_index.ntotal if self.vector_index else 0,
            "cache_size": len(self.search_cache),
            "embedding_dimension": self.embedding_dim,
            "jurisdiction": self.jurisdiction
        }
    
    def clear_cache(self) -> None:
        """Clear search cache."""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    async def export_documents(
        self,
        output_path: str,
        document_types: Optional[List[DocumentType]] = None,
        format: str = "json"
    ) -> bool:
        """
        Export documents to file.
        
        Args:
            output_path: Output file path
            document_types: Filter by document types
            format: Export format (json, csv)
            
        Returns:
            True if successful
        """
        try:
            # Build query
            sql_conditions = []
            params = []
            
            if document_types:
                type_conditions = " OR ".join(["document_type = ?"] * len(document_types))
                sql_conditions.append(f"({type_conditions})")
                params.extend([dt.value for dt in document_types])
            
            where_clause = " AND ".join(sql_conditions) if sql_conditions else "1=1"
            sql_query = f"SELECT * FROM legal_documents WHERE {where_clause}"
            
            # Execute query
            async with aiosqlite.connect(self.db_path_async) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql_query, params) as cursor:
                    rows = await cursor.fetchall()
            
            # Convert to documents
            documents = [self._row_to_document(row) for row in rows]
            
            # Export based on format
            if format.lower() == "json":
                export_data = []
                for doc in documents:
                    doc_dict = asdict(doc)
                    # Convert non-serializable fields
                    if doc_dict['embedding'] is not None:
                        doc_dict['embedding'] = doc_dict['embedding'].tolist()
                    if doc_dict['date_decided']:
                        doc_dict['date_decided'] = doc_dict['date_decided'].isoformat()
                    if doc_dict['created_at']:
                        doc_dict['created_at'] = doc_dict['created_at'].isoformat()
                    if doc_dict['updated_at']:
                        doc_dict['updated_at'] = doc_dict['updated_at'].isoformat()
                    doc_dict['document_type'] = doc_dict['document_type'].value
                    export_data.append(doc_dict)
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == "csv":
                # Convert to DataFrame
                export_data = []
                for doc in documents:
                    doc_dict = {
                        'id': doc.id,
                        'title': doc.title,
                        'document_type': doc.document_type.value,
                        'jurisdiction': doc.jurisdiction,
                        'court': doc.court,
                        'date_decided': doc.date_decided.isoformat() if doc.date_decided else None,
                        'citation': doc.citation,
                        'summary': doc.summary,
                        'precedent_value': doc.precedent_value
                    }
                    export_data.append(doc_dict)
                
                df = pd.DataFrame(export_data)
                df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(documents)} documents to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export documents: {e}")
            return False
    
    async def import_documents(
        self,
        input_path: str,
        format: str = "json",
        batch_size: int = 100
    ) -> int:
        """
        Import documents from file.
        
        Args:
            input_path: Input file path
            format: Import format (json, csv)
            batch_size: Number of documents to process in each batch
            
        Returns:
            Number of documents imported
        """
        try:
            imported_count = 0
            
            if format.lower() == "json":
                with open(input_path, 'r') as f:
                    data = json.load(f)
                
                # Process in batches
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    
                    for doc_dict in batch:
                        # Convert back to LegalDocument
                        if doc_dict.get('embedding'):
                            doc_dict['embedding'] = np.array(doc_dict['embedding'])
                        if doc_dict.get('date_decided'):
                            doc_dict['date_decided'] = datetime.fromisoformat(doc_dict['date_decided']).date()
                        if doc_dict.get('created_at'):
                            doc_dict['created_at'] = datetime.fromisoformat(doc_dict['created_at'])
                        if doc_dict.get('updated_at'):
                            doc_dict['updated_at'] = datetime.fromisoformat(doc_dict['updated_at'])
                        
                        doc_dict['document_type'] = DocumentType(doc_dict['document_type'])
                        
                        document = LegalDocument(**doc_dict)
                        
                        if await self.add_document(document):
                            imported_count += 1
            
            elif format.lower() == "csv":
                df = pd.read_csv(input_path)
                
                for _, row in df.iterrows():
                    # Create basic document from CSV
                    document = LegalDocument(
                        id=row['id'],
                        title=row['title'],
                        content=row.get('content', ''),
                        document_type=DocumentType(row['document_type']),
                        jurisdiction=row['jurisdiction'],
                        court=row.get('court'),
                        date_decided=datetime.fromisoformat(row['date_decided']).date() if pd.notna(row.get('date_decided')) else None,
                        citation=row.get('citation'),
                        summary=row.get('summary'),
                        precedent_value=row.get('precedent_value')
                    )
                    
                    if await self.add_document(document):
                        imported_count += 1
            
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            logger.info(f"Imported {imported_count} documents from {input_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import documents: {e}")
            return 0
    
    async def close(self) -> None:
        """Close knowledge base and save state."""
        # Save vector index
        self.save_vector_index()
        
        # Close database connections would be handled by aiosqlite automatically
        
        logger.info("Legal knowledge base closed")
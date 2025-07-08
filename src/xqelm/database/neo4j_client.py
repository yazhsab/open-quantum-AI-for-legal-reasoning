"""
Neo4j Graph Database Client

Neo4j client for managing legal knowledge graphs and relationships.

Copyright 2024 XQELM Research Team
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError
from loguru import logger

from ..utils.config import Neo4jConfig


class Neo4jClient:
    """
    Asynchronous Neo4j client for legal knowledge graph operations.
    """
    
    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j client.
        
        Args:
            config: Neo4j configuration
        """
        self.config = config
        self.driver = None
        self._session_pool = []
        
        logger.info(f"Initializing Neo4j client for {config.uri}")
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_timeout
            )
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            
            # Create constraints and indexes
            await self._create_constraints_and_indexes()
            
            logger.info("Successfully connected to Neo4j database")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close connection to Neo4j database."""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j database")
    
    @asynccontextmanager
    async def session(self, **kwargs):
        """Get async session context manager."""
        session = self.driver.session(**kwargs)
        try:
            yield session
        finally:
            await session.close()
    
    async def _create_constraints_and_indexes(self) -> None:
        """Create necessary constraints and indexes."""
        constraints_and_indexes = [
            # Constraints
            "CREATE CONSTRAINT legal_case_id IF NOT EXISTS FOR (c:LegalCase) REQUIRE c.case_id IS UNIQUE",
            "CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (d:LegalDocument) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT legal_entity_id IF NOT EXISTS FOR (e:LegalEntity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT statute_id IF NOT EXISTS FOR (s:Statute) REQUIRE s.statute_id IS UNIQUE",
            "CREATE CONSTRAINT precedent_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.precedent_id IS UNIQUE",
            "CREATE CONSTRAINT court_id IF NOT EXISTS FOR (c:Court) REQUIRE c.court_id IS UNIQUE",
            "CREATE CONSTRAINT judge_id IF NOT EXISTS FOR (j:Judge) REQUIRE j.judge_id IS UNIQUE",
            "CREATE CONSTRAINT lawyer_id IF NOT EXISTS FOR (l:Lawyer) REQUIRE l.lawyer_id IS UNIQUE",
            
            # Indexes
            "CREATE INDEX legal_case_type IF NOT EXISTS FOR (c:LegalCase) ON (c.case_type)",
            "CREATE INDEX legal_case_status IF NOT EXISTS FOR (c:LegalCase) ON (c.status)",
            "CREATE INDEX legal_case_date IF NOT EXISTS FOR (c:LegalCase) ON (c.filing_date)",
            "CREATE INDEX legal_document_type IF NOT EXISTS FOR (d:LegalDocument) ON (d.document_type)",
            "CREATE INDEX legal_entity_type IF NOT EXISTS FOR (e:LegalEntity) ON (e.entity_type)",
            "CREATE INDEX statute_section IF NOT EXISTS FOR (s:Statute) ON (s.section)",
            "CREATE INDEX precedent_citation IF NOT EXISTS FOR (p:Precedent) ON (p.citation)",
            "CREATE INDEX court_jurisdiction IF NOT EXISTS FOR (c:Court) ON (c.jurisdiction)",
            
            # Full-text indexes
            "CREATE FULLTEXT INDEX legal_case_text IF NOT EXISTS FOR (c:LegalCase) ON EACH [c.case_title, c.case_summary]",
            "CREATE FULLTEXT INDEX legal_document_text IF NOT EXISTS FOR (d:LegalDocument) ON EACH [d.content, d.summary]",
            "CREATE FULLTEXT INDEX statute_text IF NOT EXISTS FOR (s:Statute) ON EACH [s.title, s.content]",
            "CREATE FULLTEXT INDEX precedent_text IF NOT EXISTS FOR (p:Precedent) ON EACH [p.title, p.summary, p.judgment_text]"
        ]
        
        async with self.session() as session:
            for query in constraints_and_indexes:
                try:
                    await session.run(query)
                except Exception as e:
                    # Some constraints/indexes might already exist
                    logger.debug(f"Constraint/index creation warning: {e}")
        
        logger.info("Created Neo4j constraints and indexes")
    
    async def create_legal_case_node(
        self,
        case_id: str,
        case_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a legal case node.
        
        Args:
            case_id: Unique case identifier
            case_data: Case information
            
        Returns:
            Created node data
        """
        query = """
        CREATE (c:LegalCase {
            case_id: $case_id,
            case_number: $case_number,
            case_title: $case_title,
            case_type: $case_type,
            case_summary: $case_summary,
            status: $status,
            filing_date: $filing_date,
            court_name: $court_name,
            jurisdiction: $jurisdiction,
            parties_involved: $parties_involved,
            legal_issues: $legal_issues,
            applicable_laws: $applicable_laws,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN c
        """
        
        async with self.session() as session:
            result = await session.run(query, case_id=case_id, **case_data)
            record = await result.single()
            return dict(record["c"]) if record else None
    
    async def create_legal_document_node(
        self,
        document_id: str,
        document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a legal document node.
        
        Args:
            document_id: Unique document identifier
            document_data: Document information
            
        Returns:
            Created node data
        """
        query = """
        CREATE (d:LegalDocument {
            document_id: $document_id,
            document_name: $document_name,
            document_type: $document_type,
            content: $content,
            summary: $summary,
            file_path: $file_path,
            mime_type: $mime_type,
            file_size: $file_size,
            legal_entities: $legal_entities,
            key_phrases: $key_phrases,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN d
        """
        
        async with self.session() as session:
            result = await session.run(query, document_id=document_id, **document_data)
            record = await result.single()
            return dict(record["d"]) if record else None
    
    async def create_statute_node(
        self,
        statute_id: str,
        statute_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a statute node.
        
        Args:
            statute_id: Unique statute identifier
            statute_data: Statute information
            
        Returns:
            Created node data
        """
        query = """
        CREATE (s:Statute {
            statute_id: $statute_id,
            title: $title,
            act_name: $act_name,
            section: $section,
            subsection: $subsection,
            content: $content,
            jurisdiction: $jurisdiction,
            effective_date: $effective_date,
            amendment_history: $amendment_history,
            keywords: $keywords,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN s
        """
        
        async with self.session() as session:
            result = await session.run(query, statute_id=statute_id, **statute_data)
            record = await result.single()
            return dict(record["s"]) if record else None
    
    async def create_precedent_node(
        self,
        precedent_id: str,
        precedent_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a precedent case node.
        
        Args:
            precedent_id: Unique precedent identifier
            precedent_data: Precedent information
            
        Returns:
            Created node data
        """
        query = """
        CREATE (p:Precedent {
            precedent_id: $precedent_id,
            citation: $citation,
            title: $title,
            court_name: $court_name,
            judges: $judges,
            decision_date: $decision_date,
            summary: $summary,
            judgment_text: $judgment_text,
            legal_principles: $legal_principles,
            ratio_decidendi: $ratio_decidendi,
            obiter_dicta: $obiter_dicta,
            binding_authority: $binding_authority,
            jurisdiction: $jurisdiction,
            case_type: $case_type,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN p
        """
        
        async with self.session() as session:
            result = await session.run(query, precedent_id=precedent_id, **precedent_data)
            record = await result.single()
            return dict(record["p"]) if record else None
    
    async def create_legal_entity_node(
        self,
        entity_id: str,
        entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a legal entity node (person, organization, etc.).
        
        Args:
            entity_id: Unique entity identifier
            entity_data: Entity information
            
        Returns:
            Created node data
        """
        query = """
        CREATE (e:LegalEntity {
            entity_id: $entity_id,
            name: $name,
            entity_type: $entity_type,
            role: $role,
            description: $description,
            contact_info: $contact_info,
            jurisdiction: $jurisdiction,
            registration_number: $registration_number,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN e
        """
        
        async with self.session() as session:
            result = await session.run(query, entity_id=entity_id, **entity_data)
            record = await result.single()
            return dict(record["e"]) if record else None
    
    async def create_relationship(
        self,
        from_node_id: str,
        from_node_label: str,
        to_node_id: str,
        to_node_label: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            from_node_label: Source node label
            to_node_id: Target node ID
            to_node_label: Target node label
            relationship_type: Type of relationship
            properties: Relationship properties
            
        Returns:
            Created relationship data
        """
        properties = properties or {}
        properties.update({
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        
        # Build properties string for Cypher query
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        
        query = f"""
        MATCH (from:{from_node_label} {{{"case_id" if from_node_label == "LegalCase" else "document_id" if from_node_label == "LegalDocument" else "entity_id"}: $from_node_id}})
        MATCH (to:{to_node_label} {{{"case_id" if to_node_label == "LegalCase" else "document_id" if to_node_label == "LegalDocument" else "entity_id"}: $to_node_id}})
        CREATE (from)-[r:{relationship_type} {{{props_str}}}]->(to)
        RETURN r
        """
        
        params = {
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            **properties
        }
        
        async with self.session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return dict(record["r"]) if record else None
    
    async def find_similar_cases(
        self,
        case_type: str,
        legal_issues: List[str],
        jurisdiction: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar legal cases based on type and issues.
        
        Args:
            case_type: Type of case
            legal_issues: List of legal issues
            jurisdiction: Optional jurisdiction filter
            limit: Maximum number of results
            
        Returns:
            List of similar cases
        """
        # Build WHERE clause
        where_clauses = ["c.case_type = $case_type"]
        params = {"case_type": case_type}
        
        if jurisdiction:
            where_clauses.append("c.jurisdiction = $jurisdiction")
            params["jurisdiction"] = jurisdiction
        
        # Add legal issues filter
        if legal_issues:
            where_clauses.append("ANY(issue IN $legal_issues WHERE issue IN c.legal_issues)")
            params["legal_issues"] = legal_issues
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (c:LegalCase)
        WHERE {where_clause}
        RETURN c, 
               size([issue IN c.legal_issues WHERE issue IN $legal_issues]) as issue_overlap
        ORDER BY issue_overlap DESC, c.filing_date DESC
        LIMIT $limit
        """
        
        params["limit"] = limit
        
        async with self.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                {
                    **dict(record["c"]),
                    "similarity_score": record["issue_overlap"]
                }
                for record in records
            ]
    
    async def find_relevant_precedents(
        self,
        legal_issues: List[str],
        case_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find relevant precedents for given legal issues.
        
        Args:
            legal_issues: List of legal issues
            case_type: Optional case type filter
            jurisdiction: Optional jurisdiction filter
            limit: Maximum number of results
            
        Returns:
            List of relevant precedents
        """
        where_clauses = []
        params = {"legal_issues": legal_issues}
        
        if case_type:
            where_clauses.append("p.case_type = $case_type")
            params["case_type"] = case_type
        
        if jurisdiction:
            where_clauses.append("p.jurisdiction = $jurisdiction")
            params["jurisdiction"] = jurisdiction
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "true"
        
        query = f"""
        MATCH (p:Precedent)
        WHERE {where_clause}
        AND ANY(issue IN $legal_issues WHERE issue IN p.legal_principles)
        RETURN p,
               size([issue IN p.legal_principles WHERE issue IN $legal_issues]) as relevance_score
        ORDER BY relevance_score DESC, p.decision_date DESC
        LIMIT $limit
        """
        
        params["limit"] = limit
        
        async with self.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                {
                    **dict(record["p"]),
                    "relevance_score": record["relevance_score"]
                }
                for record in records
            ]
    
    async def find_applicable_statutes(
        self,
        legal_issues: List[str],
        jurisdiction: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find applicable statutes for given legal issues.
        
        Args:
            legal_issues: List of legal issues
            jurisdiction: Optional jurisdiction filter
            limit: Maximum number of results
            
        Returns:
            List of applicable statutes
        """
        where_clauses = []
        params = {"legal_issues": legal_issues}
        
        if jurisdiction:
            where_clauses.append("s.jurisdiction = $jurisdiction")
            params["jurisdiction"] = jurisdiction
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "true"
        
        query = f"""
        MATCH (s:Statute)
        WHERE {where_clause}
        AND ANY(issue IN $legal_issues WHERE issue IN s.keywords)
        RETURN s,
               size([issue IN s.keywords WHERE issue IN $legal_issues]) as relevance_score
        ORDER BY relevance_score DESC, s.effective_date DESC
        LIMIT $limit
        """
        
        params["limit"] = limit
        
        async with self.session() as session:
            result = await session.run(query, **params)
            records = await result.data()
            return [
                {
                    **dict(record["s"]),
                    "relevance_score": record["relevance_score"]
                }
                for record in records
            ]
    
    async def get_case_relationships(
        self,
        case_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all relationships for a legal case.
        
        Args:
            case_id: Case identifier
            relationship_types: Optional filter for relationship types
            
        Returns:
            Dictionary of relationships by type
        """
        if relationship_types:
            rel_filter = f"WHERE type(r) IN {relationship_types}"
        else:
            rel_filter = ""
        
        query = f"""
        MATCH (c:LegalCase {{case_id: $case_id}})-[r]-(n)
        {rel_filter}
        RETURN type(r) as relationship_type, r as relationship, n as related_node, labels(n) as node_labels
        """
        
        async with self.session() as session:
            result = await session.run(query, case_id=case_id)
            records = await result.data()
            
            relationships = {}
            for record in records:
                rel_type = record["relationship_type"]
                if rel_type not in relationships:
                    relationships[rel_type] = []
                
                relationships[rel_type].append({
                    "relationship": dict(record["relationship"]),
                    "related_node": dict(record["related_node"]),
                    "node_labels": record["node_labels"]
                })
            
            return relationships
    
    async def search_full_text(
        self,
        query_text: str,
        node_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search across legal documents.
        
        Args:
            query_text: Search query
            node_types: Optional filter for node types
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        results = []
        
        # Search different node types
        search_configs = [
            ("LegalCase", "legal_case_text"),
            ("LegalDocument", "legal_document_text"),
            ("Statute", "statute_text"),
            ("Precedent", "precedent_text")
        ]
        
        if node_types:
            search_configs = [
                (node_type, index) for node_type, index in search_configs
                if node_type in node_types
            ]
        
        async with self.session() as session:
            for node_type, index_name in search_configs:
                query = f"""
                CALL db.index.fulltext.queryNodes('{index_name}', $query_text)
                YIELD node, score
                RETURN node, score, '{node_type}' as node_type
                ORDER BY score DESC
                LIMIT $limit
                """
                
                result = await session.run(query, query_text=query_text, limit=limit)
                records = await result.data()
                
                for record in records:
                    results.append({
                        "node": dict(record["node"]),
                        "score": record["score"],
                        "node_type": record["node_type"]
                    })
        
        # Sort all results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get graph database statistics.
        
        Returns:
            Graph statistics
        """
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_counts": """
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """,
            "relationship_counts": """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """
        }
        
        stats = {}
        
        async with self.session() as session:
            # Get total counts
            for stat_name, query in queries.items():
                if stat_name in ["total_nodes", "total_relationships"]:
                    result = await session.run(query)
                    record = await result.single()
                    stats[stat_name] = record["count"] if record else 0
                else:
                    result = await session.run(query)
                    records = await result.data()
                    stats[stat_name] = records
        
        return stats
    
    async def delete_node(
        self,
        node_id: str,
        node_label: str,
        cascade: bool = False
    ) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: Node identifier
            node_label: Node label
            cascade: Whether to delete related nodes
            
        Returns:
            True if deleted successfully
        """
        id_field = "case_id" if node_label == "LegalCase" else "document_id" if node_label == "LegalDocument" else "entity_id"
        
        if cascade:
            query = f"""
            MATCH (n:{node_label} {{{id_field}: $node_id}})
            DETACH DELETE n
            """
        else:
            query = f"""
            MATCH (n:{node_label} {{{id_field}: $node_id}})
            WHERE NOT (n)-[]-()
            DELETE n
            """
        
        async with self.session() as session:
            result = await session.run(query, node_id=node_id)
            summary = await result.consume()
            return summary.counters.nodes_deleted > 0
    
    async def update_node(
        self,
        node_id: str,
        node_label: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a node's properties.
        
        Args:
            node_id: Node identifier
            node_label: Node label
            updates: Properties to update
            
        Returns:
            Updated node data
        """
        id_field = "case_id" if node_label == "LegalCase" else "document_id" if node_label == "LegalDocument" else "entity_id"
        
        # Build SET clause
        set_clauses = [f"n.{key} = ${key}" for key in updates.keys()]
        set_clauses.append("n.updated_at = datetime()")
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (n:{node_label} {{{id_field}: $node_id}})
        SET {set_clause}
        RETURN n
        """
        
        params = {"node_id": node_id, **updates}
        
        async with self.session() as session:
            result = await session.run(query, **params)
            record = await result.single()
            return dict(record["n"]) if record else None
    
    async def execute_custom_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query
            parameters: Query parameters
            
        Returns:
            Query results
        """
        parameters = parameters or {}
        
        async with self.session() as session:
            result = await session.run(query, **parameters)
            return await result.data()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Neo4j connection.
        
        Returns:
            Health check results
        """
        try:
            async with self.session() as session:
                result = await session.run("RETURN 1 as health_check")
                record = await result.single()
                
                if record and record["health_check"] == 1:
                    return {
                        "status": "healthy",
                        "timestamp": datetime.now().isoformat(),
                        "database": "neo4j"
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "timestamp": datetime.now().isoformat(),
                        "database": "neo4j",
                        "error": "Unexpected response"
                    }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "database": "neo4j",
                "error": str(e)
            }
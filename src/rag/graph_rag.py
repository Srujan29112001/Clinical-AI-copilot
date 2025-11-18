"""
Graph RAG for Clinical Knowledge
Uses Neo4j to store and query medical ontologies and relationships
"""

from typing import List, Dict, Optional, Any, Tuple
import logging

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available, using mock implementation")

from .vector_store import ClinicalVectorStore
from .embeddings import MedicalEmbeddings

logger = logging.getLogger(__name__)


class ClinicalGraphRAG:
    """
    Graph-based Retrieval Augmented Generation for clinical knowledge

    Features:
    - Medical ontologies (SNOMED-CT, ICD-10, RxNorm)
    - Disease-symptom-treatment relationships
    - Drug interactions and contraindications
    - Evidence-based clinical pathways
    - Integration with vector store for hybrid retrieval
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "medical_knowledge",
        vector_store: Optional[ClinicalVectorStore] = None
    ):
        """
        Initialize Graph RAG system

        Args:
            uri: Neo4j URI
            user: Username
            password: Password
            database: Database name
            vector_store: Optional vector store for hybrid search
        """
        self.database = database
        self.vector_store = vector_store

        if NEO4J_AVAILABLE:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info(f"Connected to Neo4j at {uri}")
            self._initialize_schema()
        else:
            self.driver = None
            logger.warning("Neo4j not available, using mock implementation")

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def _initialize_schema(self):
        """Initialize graph schema with medical ontology structure"""
        if not NEO4J_AVAILABLE or not self.driver:
            return

        schema_queries = [
            # Create constraints
            """
            CREATE CONSTRAINT disease_id IF NOT EXISTS
            FOR (d:Disease) REQUIRE d.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT symptom_id IF NOT EXISTS
            FOR (s:Symptom) REQUIRE s.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT treatment_id IF NOT EXISTS
            FOR (t:Treatment) REQUIRE t.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT drug_id IF NOT EXISTS
            FOR (d:Drug) REQUIRE d.id IS UNIQUE
            """,
            # Create indexes
            """
            CREATE INDEX disease_name IF NOT EXISTS
            FOR (d:Disease) ON (d.name)
            """,
            """
            CREATE INDEX symptom_name IF NOT EXISTS
            FOR (s:Symptom) ON (s.name)
            """
        ]

        with self.driver.session(database=self.database) as session:
            for query in schema_queries:
                try:
                    session.run(query)
                except Exception as e:
                    logger.debug(f"Schema query info: {e}")

        logger.info("Graph schema initialized")

    def add_disease(
        self,
        disease_id: str,
        name: str,
        snomed_id: Optional[str] = None,
        icd10: Optional[str] = None,
        category: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """Add disease node to graph"""
        query = """
        MERGE (d:Disease {id: $disease_id})
        SET d.name = $name,
            d.snomed_id = $snomed_id,
            d.icd10 = $icd10,
            d.category = $category,
            d.description = $description
        """

        self._execute_query(query, {
            'disease_id': disease_id,
            'name': name,
            'snomed_id': snomed_id,
            'icd10': icd10,
            'category': category,
            'description': description
        })

    def add_symptom(
        self,
        symptom_id: str,
        name: str,
        description: Optional[str] = None,
        eeg_marker: Optional[str] = None
    ) -> None:
        """Add symptom node to graph"""
        query = """
        MERGE (s:Symptom {id: $symptom_id})
        SET s.name = $name,
            s.description = $description,
            s.eeg_marker = $eeg_marker
        """

        self._execute_query(query, {
            'symptom_id': symptom_id,
            'name': name,
            'description': description,
            'eeg_marker': eeg_marker
        })

    def add_treatment(
        self,
        treatment_id: str,
        name: str,
        treatment_type: str,
        description: Optional[str] = None
    ) -> None:
        """Add treatment node to graph"""
        query = """
        MERGE (t:Treatment {id: $treatment_id})
        SET t.name = $name,
            t.type = $treatment_type,
            t.description = $description
        """

        self._execute_query(query, {
            'treatment_id': treatment_id,
            'name': name,
            'treatment_type': treatment_type,
            'description': description
        })

    def link_disease_symptom(
        self,
        disease_id: str,
        symptom_id: str,
        frequency: Optional[str] = None,
        severity: Optional[str] = None
    ) -> None:
        """Link disease to symptom"""
        query = """
        MATCH (d:Disease {id: $disease_id})
        MATCH (s:Symptom {id: $symptom_id})
        MERGE (d)-[r:HAS_SYMPTOM]->(s)
        SET r.frequency = $frequency,
            r.severity = $severity
        """

        self._execute_query(query, {
            'disease_id': disease_id,
            'symptom_id': symptom_id,
            'frequency': frequency,
            'severity': severity
        })

    def link_disease_treatment(
        self,
        disease_id: str,
        treatment_id: str,
        efficacy: Optional[float] = None,
        evidence_level: Optional[str] = None,
        line_of_therapy: Optional[int] = None
    ) -> None:
        """Link disease to treatment"""
        query = """
        MATCH (d:Disease {id: $disease_id})
        MATCH (t:Treatment {id: $treatment_id})
        MERGE (d)-[r:TREATED_BY]->(t)
        SET r.efficacy = $efficacy,
            r.evidence_level = $evidence_level,
            r.line_of_therapy = $line_of_therapy
        """

        self._execute_query(query, {
            'disease_id': disease_id,
            'treatment_id': treatment_id,
            'efficacy': efficacy,
            'evidence_level': evidence_level,
            'line_of_therapy': line_of_therapy
        })

    def query_diseases_by_symptoms(
        self,
        symptoms: List[str],
        eeg_markers: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Query diseases based on symptoms and EEG markers

        Args:
            symptoms: List of symptom names
            eeg_markers: Optional list of EEG markers

        Returns:
            List of matching diseases with scores
        """
        if not NEO4J_AVAILABLE or not self.driver:
            return self._mock_disease_query(symptoms)

        # Build query based on symptoms
        query = """
        MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
        WHERE s.name IN $symptoms
        WITH d, collect(s.name) as matched_symptoms, collect(r.frequency) as frequencies
        """

        # Add EEG marker filter if provided
        if eeg_markers:
            query += """
            MATCH (d)-[:HAS_SYMPTOM]->(s2:Symptom)
            WHERE s2.eeg_marker IN $eeg_markers
            WITH d, matched_symptoms, frequencies, collect(s2.eeg_marker) as matched_markers
            """

        # Calculate match score and return
        query += """
        OPTIONAL MATCH (d)-[t:TREATED_BY]->(treatment:Treatment)
        WITH d, matched_symptoms, frequencies,
             size(matched_symptoms) as match_count,
             collect(DISTINCT treatment.name) as treatments
        ORDER BY match_count DESC
        LIMIT 10
        RETURN d.id as disease_id,
               d.name as disease_name,
               d.icd10 as icd10,
               d.category as category,
               matched_symptoms,
               frequencies,
               treatments,
               match_count
        """

        params = {'symptoms': symptoms}
        if eeg_markers:
            params['eeg_markers'] = eeg_markers

        results = self._execute_query(query, params)
        return results

    def query_treatments_for_disease(
        self,
        disease_id: str,
        min_efficacy: float = 0.0
    ) -> List[Dict]:
        """
        Query recommended treatments for a disease

        Args:
            disease_id: Disease identifier
            min_efficacy: Minimum efficacy threshold

        Returns:
            List of treatments with efficacy data
        """
        query = """
        MATCH (d:Disease {id: $disease_id})-[r:TREATED_BY]->(t:Treatment)
        WHERE r.efficacy >= $min_efficacy OR r.efficacy IS NULL
        RETURN t.id as treatment_id,
               t.name as treatment_name,
               t.type as treatment_type,
               r.efficacy as efficacy,
               r.evidence_level as evidence_level,
               r.line_of_therapy as line_of_therapy
        ORDER BY r.line_of_therapy ASC, r.efficacy DESC
        """

        results = self._execute_query(query, {
            'disease_id': disease_id,
            'min_efficacy': min_efficacy
        })

        return results

    def retrieve_context(
        self,
        query: str,
        eeg_features: Optional[Dict] = None,
        top_k: int = 10
    ) -> Dict:
        """
        Retrieve comprehensive clinical context using hybrid search

        Args:
            query: Clinical query
            eeg_features: Optional EEG features dictionary
            top_k: Number of results

        Returns:
            Dictionary with graph and vector results
        """
        context = {
            'graph_results': [],
            'vector_results': [],
            'combined_score': []
        }

        # Extract symptoms from query (simplified - use NER in production)
        symptoms = self._extract_symptoms(query)

        # Extract EEG markers from features
        eeg_markers = None
        if eeg_features:
            eeg_markers = eeg_features.get('detected_patterns', [])

        # Query graph database
        if symptoms or eeg_markers:
            graph_results = self.query_diseases_by_symptoms(
                symptoms=symptoms or [],
                eeg_markers=eeg_markers
            )
            context['graph_results'] = graph_results

        # Query vector store if available
        if self.vector_store:
            vector_results = self.vector_store.search(
                query=query,
                top_k=top_k
            )
            context['vector_results'] = [
                {'text': doc.text, 'score': score, 'metadata': doc.metadata}
                for doc, score in vector_results
            ]

        return context

    def _extract_symptoms(self, query: str) -> List[str]:
        """
        Extract symptoms from query text
        Simplified version - use medical NER in production
        """
        # Common symptom keywords
        symptom_keywords = [
            'seizure', 'headache', 'dizziness', 'confusion',
            'tremor', 'weakness', 'pain', 'fever', 'fatigue'
        ]

        query_lower = query.lower()
        found_symptoms = [s for s in symptom_keywords if s in query_lower]

        return found_symptoms

    def _execute_query(self, query: str, params: Dict) -> List[Dict]:
        """Execute Cypher query"""
        if not NEO4J_AVAILABLE or not self.driver:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            return [dict(record) for record in result]

    def _mock_disease_query(self, symptoms: List[str]) -> List[Dict]:
        """Mock disease query for testing"""
        # Return mock data
        return [
            {
                'disease_id': 'epilepsy_001',
                'disease_name': 'Epilepsy',
                'icd10': 'G40',
                'category': 'Neurological',
                'matched_symptoms': symptoms[:2] if len(symptoms) >= 2 else symptoms,
                'treatments': ['Levetiracetam', 'Valproic Acid'],
                'match_count': min(len(symptoms), 2)
            }
        ]


def initialize_medical_ontology(graph_rag: ClinicalGraphRAG):
    """
    Initialize graph with basic medical ontology
    Example data - expand with real medical ontologies
    """
    # Add diseases
    graph_rag.add_disease(
        disease_id='epilepsy_001',
        name='Epilepsy',
        snomed_id='84757009',
        icd10='G40',
        category='Neurological',
        description='Chronic neurological disorder characterized by recurrent seizures'
    )

    # Add symptoms
    graph_rag.add_symptom(
        symptom_id='seizure_001',
        name='Seizure',
        description='Sudden uncontrolled electrical disturbance in the brain',
        eeg_marker='spike_wave'
    )

    graph_rag.add_symptom(
        symptom_id='confusion_001',
        name='Confusion',
        description='Altered mental state with impaired awareness'
    )

    # Add treatments
    graph_rag.add_treatment(
        treatment_id='levetiracetam_001',
        name='Levetiracetam',
        treatment_type='Antiepileptic Drug',
        description='Broad-spectrum antiepileptic medication'
    )

    # Link relationships
    graph_rag.link_disease_symptom(
        disease_id='epilepsy_001',
        symptom_id='seizure_001',
        frequency='Always',
        severity='Variable'
    )

    graph_rag.link_disease_treatment(
        disease_id='epilepsy_001',
        treatment_id='levetiracetam_001',
        efficacy=0.85,
        evidence_level='A',
        line_of_therapy=1
    )

    logger.info("Medical ontology initialized")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create graph RAG
    graph_rag = ClinicalGraphRAG(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    # Initialize with sample data
    initialize_medical_ontology(graph_rag)

    # Query example
    symptoms = ['seizure', 'confusion']
    diseases = graph_rag.query_diseases_by_symptoms(symptoms)

    print("\nDiseases matching symptoms:")
    for disease in diseases:
        print(f"\n{disease['disease_name']} (ICD-10: {disease['icd10']})")
        print(f"  Matched symptoms: {disease['matched_symptoms']}")
        print(f"  Treatments: {disease['treatments']}")

    # Close connection
    graph_rag.close()

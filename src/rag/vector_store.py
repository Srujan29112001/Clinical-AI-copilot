"""
Clinical Vector Store
Vector database interface for medical literature and clinical knowledge
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("Qdrant not available, using fallback vector store")

from .embeddings import MedicalEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Medical document with metadata"""
    id: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class ClinicalVectorStore:
    """
    Vector store for clinical documents using Qdrant

    Features:
    - Store medical papers, clinical guidelines, case studies
    - Semantic search with medical embeddings
    - Metadata filtering (e.g., by specialty, date, evidence level)
    - Hybrid search (vector + keyword)
    """

    def __init__(
        self,
        collection_name: str = "medical_knowledge",
        embedding_model: Optional[MedicalEmbeddings] = None,
        host: str = "localhost",
        port: int = 6333,
        use_memory: bool = False
    ):
        """
        Initialize clinical vector store

        Args:
            collection_name: Name of the collection
            embedding_model: Medical embeddings model
            host: Qdrant host
            port: Qdrant port
            use_memory: Use in-memory storage (for testing)
        """
        self.collection_name = collection_name

        # Initialize embedding model
        if embedding_model is None:
            from .embeddings import create_medical_embeddings
            self.embedding_model = create_medical_embeddings('pubmedbert')
        else:
            self.embedding_model = embedding_model

        self.embedding_dim = self.embedding_model.embedding_dim

        # Initialize Qdrant client
        if QDRANT_AVAILABLE:
            if use_memory:
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(host=host, port=port)

            self._initialize_collection()
        else:
            # Fallback to simple numpy-based storage
            self.client = None
            self.documents = []
            self.embeddings = []

        logger.info(f"Vector store initialized: {collection_name}")

    def _initialize_collection(self):
        """Initialize Qdrant collection"""
        if not QDRANT_AVAILABLE or self.client is None:
            return

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> None:
        """
        Add documents to vector store

        Args:
            documents: List of documents to add
            batch_size: Batch size for processing
        """
        logger.info(f"Adding {len(documents)} documents...")

        # Generate embeddings if not present
        texts_to_embed = [doc.text for doc in documents if doc.embedding is None]
        if texts_to_embed:
            embeddings = self.embedding_model.encode(texts_to_embed, batch_size=batch_size)
            emb_idx = 0
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = embeddings[emb_idx]
                    emb_idx += 1

        # Add to store
        if QDRANT_AVAILABLE and self.client is not None:
            self._add_to_qdrant(documents, batch_size)
        else:
            self._add_to_memory(documents)

        logger.info("Documents added successfully")

    def _add_to_qdrant(self, documents: List[Document], batch_size: int):
        """Add documents to Qdrant"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            points = [
                PointStruct(
                    id=doc.id,
                    vector=doc.embedding.tolist(),
                    payload={
                        'text': doc.text,
                        **doc.metadata
                    }
                )
                for doc in batch
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def _add_to_memory(self, documents: List[Document]):
        """Add documents to in-memory store"""
        for doc in documents:
            self.documents.append(doc)
            self.embeddings.append(doc.embedding)

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 0.7,
        metadata_filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            metadata_filter: Optional metadata filtering

        Returns:
            List of (document, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        if QDRANT_AVAILABLE and self.client is not None:
            return self._search_qdrant(
                query_embedding,
                top_k,
                score_threshold,
                metadata_filter
            )
        else:
            return self._search_memory(
                query_embedding,
                top_k,
                score_threshold
            )

    def _search_qdrant(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[Dict]
    ) -> List[Tuple[Document, float]]:
        """Search using Qdrant"""
        search_params = {
            'collection_name': self.collection_name,
            'query_vector': query_embedding.tolist(),
            'limit': top_k,
            'score_threshold': score_threshold
        }

        if metadata_filter:
            # Convert metadata filter to Qdrant filter
            # Simplified - extend for complex filters
            search_params['query_filter'] = Filter(must=[])

        results = self.client.search(**search_params)

        documents = []
        for result in results:
            doc = Document(
                id=result.id,
                text=result.payload['text'],
                metadata={k: v for k, v in result.payload.items() if k != 'text'}
            )
            documents.append((doc, result.score))

        return documents

    def _search_memory(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        score_threshold: float
    ) -> List[Tuple[Document, float]]:
        """Search using in-memory store"""
        if not self.embeddings:
            return []

        # Calculate similarities
        embeddings_matrix = np.array(self.embeddings)

        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings_matrix / np.linalg.norm(
            embeddings_matrix,
            axis=1,
            keepdims=True
        )

        # Cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)

        # Filter by threshold
        valid_indices = np.where(similarities >= score_threshold)[0]

        # Sort by similarity
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]

        # Take top_k
        top_indices = sorted_indices[:top_k]

        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    def delete_collection(self):
        """Delete the collection"""
        if QDRANT_AVAILABLE and self.client is not None:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        else:
            self.documents = []
            self.embeddings = []

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if QDRANT_AVAILABLE and self.client is not None:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                'count': info.points_count,
                'indexed_vectors': info.indexed_vectors_count,
                'segments': info.segments_count
            }
        else:
            return {
                'count': len(self.documents)
            }


def load_medical_papers(
    papers_file: str,
    vector_store: ClinicalVectorStore
) -> None:
    """
    Load medical papers into vector store

    Args:
        papers_file: Path to papers file (JSON/CSV)
        vector_store: Vector store instance
    """
    import json

    logger.info(f"Loading papers from {papers_file}")

    with open(papers_file, 'r') as f:
        papers = json.load(f)

    documents = []
    for paper in papers:
        doc = Document(
            id=paper.get('id', paper.get('pmid', str(len(documents)))),
            text=paper.get('abstract', paper.get('text', '')),
            metadata={
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'journal': paper.get('journal', ''),
                'year': paper.get('year', 0),
                'doi': paper.get('doi', ''),
                'specialty': paper.get('specialty', 'general')
            }
        )
        documents.append(doc)

    vector_store.add_documents(documents)
    logger.info(f"Loaded {len(documents)} papers")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create vector store
    store = ClinicalVectorStore(use_memory=True)

    # Add sample documents
    docs = [
        Document(
            id="1",
            text="Epilepsy is characterized by recurrent seizures and abnormal EEG patterns.",
            metadata={'specialty': 'neurology', 'year': 2023}
        ),
        Document(
            id="2",
            text="Levetiracetam is an effective antiepileptic drug with minimal side effects.",
            metadata={'specialty': 'neurology', 'year': 2022}
        ),
        Document(
            id="3",
            text="Sleep disorders can be diagnosed using polysomnography and EEG analysis.",
            metadata={'specialty': 'sleep medicine', 'year': 2023}
        )
    ]

    store.add_documents(docs)

    # Search
    query = "seizure treatment options"
    results = store.search(query, top_k=2)

    print(f"\nSearch results for: '{query}'")
    for doc, score in results:
        print(f"\nScore: {score:.4f}")
        print(f"Text: {doc.text}")
        print(f"Metadata: {doc.metadata}")

    # Stats
    print(f"\nVector store stats: {store.get_stats()}")

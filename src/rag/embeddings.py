"""
Medical Embeddings Module
Generate embeddings for clinical text using BioBERT and other medical models
"""

import torch
import numpy as np
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class MedicalEmbeddings:
    """
    Generate embeddings for medical/clinical text

    Supports multiple medical language models:
    - BioBERT
    - PubMedBERT
    - ClinicalBERT
    - SapBERT (for medical entity linking)
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize medical embeddings model

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu')
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        logger.info(f"Loading medical embedding model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size

        logger.info(f"Model loaded on {self.device}, embedding dim: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            pooling: Pooling strategy ('mean', 'max', 'cls')

        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Pool embeddings
            if pooling == 'mean':
                # Mean pooling with attention mask
                attention_mask = inputs['attention_mask']
                embeddings_batch = self._mean_pooling(
                    outputs.last_hidden_state,
                    attention_mask
                )
            elif pooling == 'max':
                embeddings_batch = torch.max(outputs.last_hidden_state, dim=1)[0]
            elif pooling == 'cls':
                embeddings_batch = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            # Normalize if requested
            if normalize:
                embeddings_batch = torch.nn.functional.normalize(
                    embeddings_batch,
                    p=2,
                    dim=1
                )

            embeddings.append(embeddings_batch.cpu().numpy())

        # Concatenate all batches
        embeddings = np.vstack(embeddings)

        # Return single embedding if input was single text
        if single_text:
            return embeddings[0]

        return embeddings

    def _mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling with attention mask

        Args:
            token_embeddings: Token embeddings (batch, seq_len, hidden_size)
            attention_mask: Attention mask (batch, seq_len)

        Returns:
            Pooled embeddings (batch, hidden_size)
        """
        # Expand attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Sum mask
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Mean pooling
        return sum_embeddings / sum_mask

    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Calculate cosine similarity between two texts or embeddings

        Args:
            text1: Text or embedding
            text2: Text or embedding

        Returns:
            Cosine similarity score
        """
        # Encode if necessary
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2

        # Calculate cosine similarity
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def batch_similarity(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[List[str], np.ndarray]
    ) -> np.ndarray:
        """
        Calculate similarity between query and multiple candidates

        Args:
            query: Query text or embedding
            candidates: List of candidate texts or embeddings

        Returns:
            Array of similarity scores
        """
        # Encode query
        if isinstance(query, str):
            query_emb = self.encode(query)
        else:
            query_emb = query

        # Encode candidates
        if isinstance(candidates[0], str):
            candidate_embs = self.encode(candidates)
        else:
            candidate_embs = np.array(candidates)

        # Calculate similarities
        # Normalize embeddings
        query_emb = query_emb / np.linalg.norm(query_emb)
        candidate_embs = candidate_embs / np.linalg.norm(
            candidate_embs,
            axis=1,
            keepdims=True
        )

        # Cosine similarity
        similarities = np.dot(candidate_embs, query_emb)

        return similarities


class MedicalEntityEmbeddings(MedicalEmbeddings):
    """
    Specialized embeddings for medical entities (diseases, drugs, symptoms)
    Uses SapBERT or similar entity-focused models
    """

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: Optional[str] = None
    ):
        super().__init__(model_name=model_name, device=device)

    def encode_entity(
        self,
        entity: str,
        entity_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Encode medical entity with optional type information

        Args:
            entity: Entity text (e.g., "epilepsy", "levetiracetam")
            entity_type: Optional entity type (e.g., "disease", "drug")

        Returns:
            Entity embedding
        """
        # Add type information if provided
        if entity_type:
            text = f"[{entity_type.upper()}] {entity}"
        else:
            text = entity

        return self.encode(text)


def create_medical_embeddings(
    model_type: str = 'pubmedbert',
    device: Optional[str] = None
) -> MedicalEmbeddings:
    """
    Factory function to create medical embeddings

    Args:
        model_type: Type of model ('pubmedbert', 'biobert', 'clinicalbert', 'sapbert')
        device: Device to use

    Returns:
        MedicalEmbeddings instance
    """
    model_map = {
        'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'biobert': 'dmis-lab/biobert-v1.1',
        'clinicalbert': 'emilyalsentzer/Bio_ClinicalBERT',
        'sapbert': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
    }

    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_map.keys())}")

    model_name = model_map[model_type]

    if model_type == 'sapbert':
        return MedicalEntityEmbeddings(model_name=model_name, device=device)
    else:
        return MedicalEmbeddings(model_name=model_name, device=device)


if __name__ == "__main__":
    # Test medical embeddings
    logging.basicConfig(level=logging.INFO)

    embeddings = create_medical_embeddings('pubmedbert')

    # Test texts
    texts = [
        "The patient presents with seizures and abnormal EEG patterns.",
        "Treatment with antiepileptic drugs was initiated.",
        "Sleep disorders are common in neurological patients."
    ]

    # Encode
    print("Encoding texts...")
    embs = embeddings.encode(texts)
    print(f"Embeddings shape: {embs.shape}")

    # Calculate similarities
    query = "epilepsy seizure treatment"
    similarities = embeddings.batch_similarity(query, texts)

    print("\nSimilarities:")
    for text, sim in zip(texts, similarities):
        print(f"  {sim:.4f}: {text}")

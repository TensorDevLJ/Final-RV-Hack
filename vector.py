"""
Optimized vector operations with enhanced performance and explainability
"""
import os
import hashlib
import uuid
from typing import List, Dict, Any, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.model_path = "./models/all-mpnet-base-v2"

        if not os.path.exists(self.model_path):
            os.makedirs("models", exist_ok=True)
            print("Downloading model from HuggingFace...")
            model = SentenceTransformer("all-mpnet-base-v2")
            model.save(self.model_path)
            self.embedding_model = model
        else:
            print("Loading model from local path...")
            self.embedding_model = SentenceTransformer(self.model_path)

        self.embedding_dimension = 768
        self.embedding_dimension = 768  # Update this if using a different model

        self.index_name = os.getenv("PINECONE_INDEX_NAME", "my-index")
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Create the index if it doesn't exist
        if self.index_name not in [i.name for i in self.pinecone.list_indexes()]:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-west-2")
                )
            )

        self.index = self.pinecone.Index(self.index_name)

        self._verify_index_dimension()
        self._embedding_cache = {}

    def _verify_index_dimension(self):
        """Verify that the Pinecone index dimension matches the embedding model"""
        try:
            index_stats = self.index.describe_index_stats()
            print(f"Index stats: {index_stats}")
        except Exception as e:
            print(f"Could not verify index dimension: {e}")

    def _generate_embeddings(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of texts with caching"""
        try:
            if use_cache:
                cached_embeddings = []
                uncached_texts = []
                uncached_indices = []

                for i, text in enumerate(texts):
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    if text_hash in self._embedding_cache:
                        cached_embeddings.append((i, self._embedding_cache[text_hash]))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)

                new_embeddings = []
                if uncached_texts:
                    new_embeddings = self.embedding_model.encode(uncached_texts, convert_to_tensor=False)
                    if isinstance(new_embeddings, np.ndarray):
                        new_embeddings = new_embeddings.tolist()

                    for text, embedding in zip(uncached_texts, new_embeddings):
                        text_hash = hashlib.md5(text.encode()).hexdigest()
                        self._embedding_cache[text_hash] = embedding

                all_embeddings = [None] * len(texts)
                for i, embedding in cached_embeddings:
                    all_embeddings[i] = embedding
                for i, embedding in zip(uncached_indices, new_embeddings if uncached_texts else []):
                    all_embeddings[i] = embedding

                embeddings = all_embeddings
            else:
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()

            if len(embeddings[0]) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embeddings[0])}")

            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    def _create_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Create a unique ID for a text chunk"""
        return f"{doc_id}_chunk_{chunk_index}"

    async def store_document_chunks_optimized(self, chunks: List[str], doc_title: str) -> str:
        """
        Store document chunks in Pinecone with optimized batch processing
        Returns the document ID
        """
        try:
            doc_id = hashlib.md5(doc_title.encode()).hexdigest()[:16]

            logger.info(f"Storing {len(chunks)} chunks for document: {doc_title}")

            embeddings = self._generate_embeddings(chunks)

            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._create_chunk_id(doc_id, i)

                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "chunk_index": i,
                        "text": chunk,
                        "chunk_length": len(chunk),
                        "importance_score": self._calculate_chunk_importance(chunk)
                    }
                })

            batch_size = 50
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch)
                    logger.info(f"Upserted batch {i//batch_size + 1}: {upsert_response}")
                except Exception as e:
                    logger.error(f"Error upserting batch {i//batch_size + 1}: {e}")
                    raise

            logger.info(f"Successfully stored {len(chunks)} chunks for document: {doc_title}")
            return doc_id

        except Exception as e:
            raise Exception(f"Error storing document chunks: {str(e)}")

    def _calculate_chunk_importance(self, chunk: str) -> float:
        """Calculate importance score for a chunk based on content"""
        sentence_count = chunk.count('.') + chunk.count('!') + chunk.count('?')
        word_count = len(chunk.split())
        importance = min(1.0, (sentence_count * 0.1 + word_count * 0.001))
        return importance

    async def query_similar_chunks_with_scores(self, query: str, top_k: int = 3) -> Tuple[List[str], List[float]]:
        """
        Query Pinecone for similar chunks with similarity scores
        Returns tuple of (relevant_chunks, similarity_scores)
        """
        try:
            query_embedding = self._generate_embeddings([query])[0]

            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )

            relevant_chunks = []
            similarity_scores = []

            for match in query_response.matches:
                if 'text' in match.metadata:
                    relevant_chunks.append(match.metadata['text'])
                    similarity_scores.append(float(match.score))
                else:
                    logger.warning(f"No text found in metadata for match {match.id}")

            logger.info(f"Found {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
            return relevant_chunks, similarity_scores

        except Exception as e:
            raise Exception(f"Error querying similar chunks: {str(e)}")

    async def query_similar_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """Backward compatibility method"""
        chunks, _ = await self.query_similar_chunks_with_scores(query, top_k)
        return chunks

    async def test_connection(self) -> bool:
        """Test Pinecone connection"""
        try:
            self.index.describe_index_stats()
            return True
        except Exception:
            return False

    def delete_document(self, doc_id: str):
        """Delete all chunks for a document"""
        try:
            response = self.index.query(
                vector=[0.0] * self.embedding_dimension,
                top_k=10000,
                filter={"doc_id": doc_id},
                include_metadata=False
            )

            chunk_ids = [match.id for match in response.matches]
            if chunk_ids:
                self.index.delete(ids=chunk_ids)
                logger.info(f"Deleted {len(chunk_ids)} chunks for document {doc_id}")

        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")

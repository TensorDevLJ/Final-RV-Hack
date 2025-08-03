"""
Vector operations using Pinecone and SentenceTransformer embeddings
"""
import os
import hashlib
import uuid
from typing import List, Dict, Any
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self):
        """Initialize Pinecone and SentenceTransformer"""
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not all([api_key, environment, index_name]):
            raise ValueError("Missing Pinecone configuration in environment variables")
        
        self.pinecone = Pinecone(api_key=api_key)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.embedding_dimension = 768  # Dimension for paraphrase-mpnet-base-v2
        
        # Connect to index
        self.index_name = index_name
        self.index = self.pinecone.Index(index_name)
        
        # Verify index dimension matches model
        self._verify_index_dimension()
    
    def _verify_index_dimension(self):
        """Verify that the Pinecone index dimension matches the embedding model"""
        try:
            index_stats = self.index.describe_index_stats()
            # If index is empty, we can't check dimension directly
            # We'll check when we first upsert
            print(f"Index stats: {index_stats}")
        except Exception as e:
            print(f"Could not verify index dimension: {e}")
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            # Ensure embeddings are in the correct format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Verify dimension
            if len(embeddings[0]) != self.embedding_dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dimension}, got {len(embeddings[0])}")
            
            return embeddings
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def _create_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Create a unique ID for a text chunk"""
        return f"{doc_id}_chunk_{chunk_index}"
    
    async def store_document_chunks(self, chunks: List[str], doc_title: str) -> str:
        """
        Store document chunks in Pinecone with embeddings
        Returns the document ID
        """
        try:
            # Generate document ID
            doc_id = hashlib.md5(doc_title.encode()).hexdigest()[:16]
            
            # Generate embeddings for all chunks
            embeddings = self._generate_embeddings(chunks)
            
            # Prepare vectors for upsert
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._create_chunk_id(doc_id, i)
                
                # Ensure embedding is a list of floats
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
                        "chunk_length": len(chunk)
                    }
                })
            
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    upsert_response = self.index.upsert(vectors=batch)
                    print(f"Upserted batch {i//batch_size + 1}: {upsert_response}")
                except Exception as e:
                    print(f"Error upserting batch {i//batch_size + 1}: {e}")
                    raise
            
            print(f"Successfully stored {len(chunks)} chunks for document: {doc_title}")
            return doc_id
            
        except Exception as e:
            raise Exception(f"Error storing document chunks: {str(e)}")
    
    async def query_similar_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Query Pinecone for similar chunks based on the question
        Returns list of relevant text chunks
        """
        try:
            # Generate embedding for the query
            query_embedding = self._generate_embeddings([query])[0]
            
            # Ensure query embedding is a list of floats
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Extract text chunks from results
            relevant_chunks = []
            for match in query_response.matches:
                if 'text' in match.metadata:
                    relevant_chunks.append(match.metadata['text'])
                else:
                    print(f"Warning: No text found in metadata for match {match.id}")
            
            print(f"Found {len(relevant_chunks)} relevant chunks for query: {query[:50]}...")
            return relevant_chunks
            
        except Exception as e:
            raise Exception(f"Error querying similar chunks: {str(e)}")
    
    def delete_document(self, doc_id: str):
        """Delete all chunks for a document"""
        try:
            # Query for all chunks with this doc_id
            query_response = self.index.query(
                vector=[0.0] * self.embedding_dimension,  # Dummy vector
                top_k=10000,  # Large number to get all
                filter={"doc_id": doc_id},
                include_metadata=False
            )
            
            # Delete chunks
            chunk_ids = [match.id for match in query_response.matches]
            if chunk_ids:
                self.index.delete(ids=chunk_ids)
                print(f"Deleted {len(chunk_ids)} chunks for document {doc_id}")
            
        except Exception as e:
            raise Exception(f"Error deleting document: {str(e)}")
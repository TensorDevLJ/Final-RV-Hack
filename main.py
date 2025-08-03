"""
FastAPI PDF QA System
Optimized for accuracy, token efficiency, latency, and explainability
"""
import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List
import uvicorn
from dotenv import load_dotenv
import logging

from vector import VectorStore
from llm import LLMClient
from utils import PDFProcessor, TextChunker
from auth import verify_bearer_token

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="PDF QA System",
    description="Optimized RAG system for document Q&A with explainability",
    version="1.0.0"
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
vector_store = VectorStore()
llm_client = LLMClient()
pdf_processor = PDFProcessor()
text_chunker = TextChunker()

class PDFQARequest(BaseModel):
    pdf_url: HttpUrl = Field(..., description="PDF document URL")
    questions: List[str]

class PDFQAResponse(BaseModel):
    answers: List[str] = Field(..., description="Direct answers to questions")
    metadata: dict = Field(default_factory=dict, description="Processing metadata")
    explainability: dict = Field(default_factory=dict, description="Answer reasoning")

@app.post("/hackrx/run")
async def process_pdf_qa(
    request: PDFQARequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Main endpoint for PDF QA processing with optimized performance
    """
    start_time = time.time()

    # Verify authentication
    verify_bearer_token(credentials.credentials)

    try:
        logger.info(f"Processing PDF: {request.pdf_url}")

        # Step 1: Extract text from PDF
        pdf_text, pdf_title = await pdf_processor.extract_text_from_url(str(request.pdf_url))

        if not pdf_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Step 2: Split text into chunks
        chunks = text_chunker.split_text_optimized(pdf_text)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated from PDF")

        # Step 3: Generate embeddings and store in Pinecone
        doc_id = await vector_store.store_document_chunks_optimized(chunks, pdf_title)

        # Step 4: Process each question
        answers = []
        explainability_data = {}

        for question in request.questions:
            try:
                # Query relevant chunks with scoring
                relevant_chunks, chunk_scores = await vector_store.query_similar_chunks_with_scores(
                    question, top_k=3  # Reduced for token efficiency
                )

                answer, reasoning = await llm_client.generate_answer_optimized(
                    question, relevant_chunks
                )

                # Clean the answer prefix if present
                if isinstance(answer, str) and answer.strip().lower().startswith("answer:"):
                    answer_clean = answer.split("ANSWER:", 1)[-1].strip()
                else:
                    answer_clean = answer.strip() if answer else "I cannot find sufficient information to answer this question."

                if not answer_clean:
                    answer_clean = "I cannot find sufficient information to answer this question."
                if len(answer_clean) > 500:
                    answer_clean = answer_clean[:500] + "..."

                answers.append(answer_clean)

                explainability_data[question] = {
                    "relevant_chunks": len(relevant_chunks),
                    "chunk_scores": chunk_scores,
                    "reasoning": reasoning or "No reasoning provided",
                    "confidence": "high" if max(chunk_scores) > 0.8 else "medium"
                }

            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append("I cannot find sufficient information to answer this question.")
                explainability_data[question] = {
                    "error": str(e),
                    "confidence": "error"
                }

        processing_time = time.time() - start_time

        return {
            "answers": answers,
            "metadata": {
                "pdf_title": pdf_title,
                "chunks_processed": len(chunks),
                "processing_time_seconds": round(processing_time, 2),
                "questions_processed": len(request.questions)
            },
            "explainability": explainability_data
        }

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/v1/health")
async def health_check():
    """Enhanced health check with component status"""
    try:
        # Test LLM connection
        llm_status = llm_client.test_connection()

        # Test vector store connection
        vector_status = await vector_store.test_connection()

        return {
            "status": "healthy" if llm_status and vector_status else "degraded",
            "components": {
                "llm": "healthy" if llm_status else "unhealthy",
                "vector_store": "healthy" if vector_status else "unhealthy"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "version": "1.0.0"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Optimized PDF QA System API",
        "docs": "/docs",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/api/v1/health"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

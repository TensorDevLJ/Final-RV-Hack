
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn
from dotenv import load_dotenv

from vector import VectorStore
from llm import LLMClient
from utils import PDFProcessor, TextChunker

# Load environment variables
load_dotenv()

app = FastAPI(
    title="PDF QA System",
    description="A system that answers questions based on PDF content using RAG",
    version="1.0.0"
)

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
    pdf_url: HttpUrl
    questions: List[str]
    
class PDFQAResponse(BaseModel):
    answers: List[dict]
    pdf_title: str
    chunks_processed: int

@app.post("/hackrx/run", response_model=PDFQAResponse)
async def process_pdf_qa(request: PDFQARequest):
    """
    Main endpoint for PDF QA processing
    """
    try:
        # Step 1: Extract text from PDF
        pdf_text, pdf_title = await pdf_processor.extract_text_from_url(str(request.pdf_url))
        
        if not pdf_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Step 2: Split text into chunks
        chunks = text_chunker.split_text(pdf_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated from PDF")
        
        # Step 3: Generate embeddings and store in Pinecone
        doc_id = await vector_store.store_document_chunks(chunks, pdf_title)
        
        # Step 4: Process each question
        answers = []
        for question in request.questions:
            try:
                # Query relevant chunks
                relevant_chunks = await vector_store.query_similar_chunks(question, top_k=5)
                
                # Generate answer using LLM
                answer = await llm_client.generate_answer(question, relevant_chunks)
                
                answers.append({
                    "question": question,
                    "answer": answer,
                    "confidence": "high" if len(relevant_chunks) >= 3 else "medium"
                })
            except Exception as e:
                answers.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "confidence": "error"
                })
        
        return PDFQAResponse(
            answers=answers,
            pdf_title=pdf_title,
            chunks_processed=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "PDF QA System API", "docs": "/docs"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
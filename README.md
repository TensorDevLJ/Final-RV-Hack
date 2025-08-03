
# Final-RV-Hack
=======
# FastAPI PDF QA System

A production-ready FastAPI system that processes PDF documents and answers questions using RAG (Retrieval-Augmented Generation).

## Features

- 📄 PDF text extraction from URLs
- 🔍 Intelligent text chunking with overlap
- 🧠 Semantic embeddings using SentenceTransformer
- 📊 Vector storage in Pinecone
- 🤖 Question answering with Groq's Llama3 model
- 🚀 Fast, async FastAPI endpoints
- 📋 Comprehensive error handling

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF URL       │    │   Text Chunks    │    │   Embeddings    │
│                 │───▶│                  │───▶│                 │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Final Answer  │    │   LLM (Groq)     │             │
│                 │◀───│                  │             │
│                 │    │                  │             │
└─────────────────┘    └──────────────────┘             │
                                  ▲                     │
                                  │                     ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Relevant Context │    │   Pinecone      │
                       │                  │◀───│   Vector DB     │
                       │                  │    │                 │
                       └──────────────────┘    └─────────────────┘
```

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Set up Pinecone index:**
   - Create a Pinecone account and project
   - Create an index with dimension **768** (matches paraphrase-mpnet-base-v2)
   - Use cosine similarity metric

4. **Run the application:**
   ```bash
   python main.py
   ```

## API Usage

### Process PDF and Answer Questions

**POST** `/hackrx/run`

```json
{
  "pdf_url": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?",
    "Who are the authors?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    {
      "question": "What is the main topic of this document?",
      "answer": "The document discusses...",
      "confidence": "high"
    }
  ],
  "pdf_title": "Research Paper Title",
  "chunks_processed": 15
}
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key for LLM | Yes |
| `PINECONE_API_KEY` | Pinecone API key | Yes |
| `PINECONE_ENVIRONMENT` | Pinecone environment (e.g., us-east-1-aws) | Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | Yes |
| `PORT` | Server port (default: 8000) | No |

### Embedding Model

- **Model:** `paraphrase-mpnet-base-v2`
- **Dimension:** 768
- **Use case:** Semantic similarity for question-answering

### LLM Model

- **Provider:** Groq
- **Model:** `llama3-8b-8192`
- **Temperature:** 0.3 (for consistent answers)

## Troubleshooting

### "Could not infer dtype of Index" Error

This error typically occurs when:
1. **Dimension mismatch:** Ensure your Pinecone index dimension is 768
2. **Empty embeddings:** Check that embeddings are properly generated
3. **Data type issues:** Ensure embeddings are lists of floats

**Solution:**
```python
# Verify embedding format before upsert
embedding = embedding_model.encode([text])[0]
if isinstance(embedding, np.ndarray):
    embedding = embedding.tolist()
```

### PDF Processing Issues

1. **Large files:** Implement file size limits (50MB recommended)
2. **Protected PDFs:** Handle password-protected documents
3. **Scanned PDFs:** Consider OCR for image-based PDFs

### Rate Limiting

- Implement exponential backoff for API calls
- Consider batch processing for large documents
- Monitor API quotas

## File Structure

```
├── main.py              # FastAPI application and routes
├── vector.py            # Pinecone vector operations
├── llm.py              # Groq LLM integration
├── utils.py            # PDF processing and text chunking
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variables template
├── .env              # Your actual environment variables
└── README.md         # This file
```

## Performance Optimization

1. **Chunking Strategy:** Optimize chunk size based on your use case
2. **Embedding Caching:** Cache embeddings for frequently accessed documents
3. **Async Operations:** All heavy operations are async for better performance
4. **Batch Processing:** Upsert vectors in batches of 100

## Security

- API keys are loaded from environment variables
- File size limits prevent abuse
- Input validation on all endpoints
- CORS configuration for web frontend integration

## Monitoring

- Health check endpoint: `GET /health`
- Detailed error messages for debugging
- Logging for tracking performance


"""
Test script for the PDF QA API
"""
import requests
import json
import asyncio

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.json()}")

def test_pdf_qa():
    """Test PDF QA endpoint"""
    payload = {
        "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf",  # Example research paper
        "questions": [
            "What is the main topic of this paper?",
            "What methodology was used?",
            "What are the key findings?"
        ]
    }
    
    print("Sending request to /hackrx/run...")
    response = requests.post(
        f"{API_BASE_URL}/hackrx/run",
        json=payload,
        timeout=60
    )
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPDF Title: {result['pdf_title']}")
        print(f"Chunks processed: {result['chunks_processed']}")
        print("\nAnswers:")
        for i, answer in enumerate(result['answers'], 1):
            print(f"\n{i}. Q: {answer['question']}")
            print(f"   A: {answer['answer'][:200]}...")
            print(f"   Confidence: {answer['confidence']}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing PDF QA API...")
    
    # Test health first
    test_health()
    print("\n" + "="*50 + "\n")
    
    # Test PDF QA
    test_pdf_qa()
"""
Comprehensive test script for the optimized PDF QA API
"""
import requests
import json
import time

API_BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "cc26c963f817b3bd6b14f3c9e45e741488cd4306c5e708722aeac3a1f3f33937"

def get_headers():
    """Get headers with bearer token"""
    return {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

def test_health():
    """Test health endpoints"""
    print("Testing health endpoints...")
    
    # Test basic health
    response = requests.get(f"{API_BASE_URL}/health", headers=get_headers())
    print(f"Health check: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test enhanced health
    response = requests.get(f"{API_BASE_URL}/api/v1/health", headers=get_headers())
    print(f"Enhanced health check: {response.status_code}")
    print(f"Response: {response.json()}")

def test_pdf_qa():
    """Test PDF QA endpoint with sample questions"""
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    print("Sending request to /hackrx/run...")
    start_time = time.time()
    
    response = requests.post(
        f"{API_BASE_URL}/hackrx/run",
        json=payload,
        headers=get_headers(),
        timeout=60
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Status code: {response.status_code}")
    print(f"Processing time: {processing_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\nMetadata:")
        print(f"  PDF Title: {result['metadata']['pdf_title']}")
        print(f"  Chunks processed: {result['metadata']['chunks_processed']}")
        print(f"  Questions processed: {result['metadata']['questions_processed']}")
        print(f"  Server processing time: {result['metadata']['processing_time_seconds']} seconds")
        
        print("\nAnswers:")
        for i, (question, answer) in enumerate(zip(payload['questions'], result['answers']), 1):
            print(f"\n{i}. Q: {question}")
            print(f"   A: {answer}")
            
            # Show explainability data if available
            if question in result.get('explainability', {}):
                exp_data = result['explainability'][question]
                print(f"   Confidence: {exp_data.get('confidence', 'unknown')}")
                print(f"   Relevant chunks: {exp_data.get('relevant_chunks', 0)}")
                if 'reasoning' in exp_data:
                    print(f"   Reasoning: {exp_data['reasoning'][:100]}...")
    else:
        print(f"Error: {response.text}")

def test_authentication():
    """Test authentication with invalid token"""
    print("\nTesting authentication...")
    
    # Test with invalid token
    invalid_headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": "https://example.com/test.pdf",
        "questions": ["Test question?"]
    }
    
    response = requests.post(
        f"{API_BASE_URL}/hackrx/run",
        json=payload,
        headers=invalid_headers,
        timeout=10
    )
    
    print(f"Invalid token test - Status: {response.status_code}")
    if response.status_code == 401:
        print("✅ Authentication working correctly")
    else:
        print("❌ Authentication issue")

if __name__ == "__main__":
    print("Testing Optimized PDF QA API...")
    print("=" * 50)
    
    # Test health first
    test_health()
    print("\n" + "=" * 50)
    
    # Test authentication
    test_authentication()
    print("\n" + "=" * 50)
    
    # Test PDF QA
    test_pdf_qa()
    
    print("\n" + "=" * 50)
    print("Testing completed!")
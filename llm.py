"""
Language Model integration using Groq API
"""
import os
import time
from typing import List
import requests
import json

class LLMClient:
    def __init__(self):
        """Initialize Groq client"""
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _create_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Create a prompt for the LLM with context and question"""
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from a PDF document.

Context from PDF:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information to answer the question, say "I cannot find sufficient information in the provided context to answer this question."
- Be concise but comprehensive in your answer
- Quote relevant parts of the context when applicable

Answer:"""
        
        return prompt

    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate answer using Groq LLM based on context chunks, with retry on rate limit (429)
        """
        if not context_chunks:
            return "I cannot find sufficient information in the provided context to answer this question."
        
        prompt = self._create_prompt(question, context_chunks)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "top_p": 0.9
        }

        retries = 3
        wait_time = 15  # seconds

        for attempt in range(retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        return response_data['choices'][0]['message']['content'].strip()
                    else:
                        return "Error: No response generated"
                
                elif response.status_code == 429:
                    print(f"[Groq Rate Limit] Attempt {attempt+1} hit 429. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue  # Retry on rate limit
                
                else:
                    return f"Groq API error: {response.status_code} - {response.text}"

            except requests.RequestException as e:
                return f"Network error: {str(e)}"
            except Exception as e:
                return f"Error generating answer: {str(e)}"

        return "Groq API rate limit exceeded after multiple retries."
    
    def test_connection(self) -> bool:
        """Test if the Groq API connection is working"""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
        except:
            return False

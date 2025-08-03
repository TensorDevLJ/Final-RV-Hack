"""
Optimized Language Model integration with token efficiency and explainability
"""
import os
from typing import List, Tuple
import requests
import logging

logger = logging.getLogger(__name__)

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
    
    def _create_optimized_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Create an optimized prompt for token efficiency and accuracy"""
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        prompt = f"""Answer the question based ONLY on the provided context. Be precise and direct.

Context:
{context}

Question: {question}

Rules:
1. Use ONLY the provided context
2. Be concise and direct
3. If insufficient information, say "I cannot find sufficient information to answer this question."
4. Quote specific parts when relevant

Answer:"""
        return prompt
    
    def _create_explainable_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Create a prompt that includes reasoning explanation"""
        context = "\n\n".join([f"Context {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)])
        prompt = f"""You are a policy analyst AI. Based only on the context provided below, answer the user's question clearly and specifically. 
If the answer requires conditions or exceptions, mention them. 
Do not make assumptions or generalizations. Cite any applicable numbers, durations, or limitations.


Context:
{context}

Question: {question}

Provide your response in this format:
ANSWER: [Your direct answer]
REASONING: [Brief explanation of how you derived the answer from the context]

Rules:
1. Use ONLY the provided context
2. If you Know  the answers correctly and accuratle answer in polite n understanble way
3. Be precise and factual
5. If the answer is not in the context, state "I cannot find sufficient information to answer this question."
4. If insufficient information, state this clearly
"""
        return prompt
    
    async def generate_answer_optimized(self, question: str, context_chunks: List[str]) -> Tuple[str, str]:
        """
        Generate answer with reasoning using optimized token usage
        Returns (answer, reasoning)
        """
        try:
            if not context_chunks:
                return "I cannot find sufficient information to answer this question.", "No relevant context found"
            
            prompt = self._create_explainable_prompt(question, context_chunks)

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 500,
                "top_p": 0.9,
                "stop": ["REASONING:", "\n\n"]
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
            
            response_data = response.json()

            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content'].strip()

                # Attempt to parse answer and reasoning
                if "ANSWER:" in content and "REASONING:" in content:
                    parts = content.split("REASONING:")
                    answer = parts[0].replace("ANSWER:", "").strip()
                    reasoning = parts[1].strip()
                else:
                    # fallback if formatting isn't followed
                    answer = content.strip()
                    reasoning = "Could not parse explicit reasoning"

                return answer, reasoning
            else:
                return "Error: No response generated", "API response error"
                
        except requests.RequestException as e:
            logger.error(f"Network error: {str(e)}")
            return f"Network error: {str(e)}", "Network connectivity issue"
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}", "LLM processing error"
    
    async def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Backward compatibility method - returns only the answer
        """
        answer, _ = await self.generate_answer_optimized(question, context_chunks)
        return answer

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

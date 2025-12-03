import os
import requests
from django.conf import settings

class LLMService:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required. Please set it in your .env file")
        
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = os.getenv('GROQ_MODEL', 'llama2-70b-4096')  # or 'mixtral-8x7b-32768'

    def generate_explanation(self, user_id, product_id, user_history, product_info):
        """Generate explanation for why a product is recommended using Groq API"""
        
        # Build context
        context = self._build_context(user_history, product_info)
        
        # Generate explanation using Groq API
        explanation = self._generate_groq(context)
        
        return explanation

    def _build_context(self, user_history, product_info):
        """Build context string for LLM"""
        
        # Format user history
        history_text = ""
        if user_history:
            if isinstance(user_history, str):
                history_text = user_history
            else:
                history_items = []
                for item in user_history[:5]:  # Last 5 interactions
                    if isinstance(item, dict):
                        history_items.append(
                            f"- {item.get('product', 'Unknown')} ({item.get('type', 'view')})"
                        )
                history_text = "\n".join(history_items) if history_items else "No recent history"
        
        context = f"""You are a helpful e-commerce assistant. Based on the user's behavior and preferences, explain why this product is recommended.

User's Recent Activity:
{history_text if history_text else "User is new or has limited interaction history"}

Product Being Recommended:
- Name: {product_info.get('name', 'Unknown Product')}
- Category: {product_info.get('category', 'General')}
- Description: {product_info.get('description', '')[:200]}
- Tags: {', '.join(product_info.get('tags', []))}

Write a clear, concise explanation (2-3 sentences) explaining why this product matches the user's interests and behavior. Be specific and helpful."""
        
        return context

    def _generate_groq(self, context):
        """Generate explanation using Groq API"""
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful e-commerce assistant that explains product recommendations in a clear, concise, and personalized manner. Keep responses to 2-3 sentences."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.groq_url, 
                json=data, 
                headers=headers, 
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                explanation = result['choices'][0]['message']['content'].strip()
                return explanation
            else:
                print(f"Groq API Error: {response.status_code} - {response.text}")
                return self._generate_fallback_explanation(context)
                
        except requests.exceptions.Timeout:
            print("Groq API request timed out")
            return self._generate_fallback_explanation(context)
        except requests.exceptions.RequestException as e:
            print(f"Groq API request failed: {e}")
            return self._generate_fallback_explanation(context)
        except Exception as e:
            print(f"Unexpected error in Groq API call: {e}")
            return self._generate_fallback_explanation(context)

    def _generate_fallback_explanation(self, context):
        """Fallback explanation if Groq API fails"""
        # Extract basic info from context for fallback
        if "Category:" in context:
            category = context.split("Category:")[1].split("\n")[0].strip()
            return f"Based on your browsing history and interest in {category}, we recommend this product as it aligns with your preferences and previous interactions."
        return "This product is recommended based on your browsing and purchase history, matching your preferences and interests."


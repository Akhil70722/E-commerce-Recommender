import os
import requests
import time
from django.conf import settings
from django.core.cache import cache


class LLMService:
    """
    Enhanced LLM Service with Groq API integration
    Features:
    - Better prompts for personalized explanations
    - Batch processing support
    - Caching for performance
    - Retry logic with exponential backoff
    - Rate limiting handling
    """
    
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required. Please set it in your .env file")
        
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        # Valid Groq models: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it
        self.model = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')  # Fast and reliable default
        self.max_retries = 3
        self.timeout = 20
        
        # Validate model name
        valid_models = [
            'llama-3.1-70b-versatile',
            'llama-3.1-8b-instant',
            'llama-3.2-3b-instant',
            'mixtral-8x7b-32768',
            'gemma2-9b-it',
            'llama-3.3-70b-versatile'
        ]
        
        if self.model not in valid_models:
            print(f"Warning: Model '{self.model}' may not be valid. Using 'llama-3.1-8b-instant' instead.")
            self.model = 'llama-3.1-8b-instant'
        
    def generate_explanation(self, user_id, product_id, user_history, product_info, recommendation_score=None):
        """
        Generate explanation for why a product is recommended using Groq API
        
        Args:
            user_id: User ID
            product_id: Product ID
            user_history: List of user's recent interactions
            product_info: Dictionary with product information
            recommendation_score: Optional recommendation score for context
        """
        # Check cache first
        cache_key = f"explanation_{user_id}_{product_id}"
        cached = cache.get(cache_key)
        if cached:
            return cached
        
        # Build enhanced context
        context = self._build_enhanced_context(user_history, product_info, recommendation_score)
        
        # Generate explanation using Groq API with retry logic
        explanation = self._generate_groq_with_retry(context)
        
        # Cache the explanation for 1 hour
        if explanation:
            cache.set(cache_key, explanation, timeout=3600)
        
        return explanation
    
    def _build_enhanced_context(self, user_history, product_info, recommendation_score=None):
        """Build enhanced context string for LLM with more details"""
        
        # Format user history with more details
        history_text = ""
        if user_history:
            if isinstance(user_history, str):
                history_text = user_history
            else:
                history_items = []
                for item in user_history[:10]:  # Last 10 interactions for better context
                    if isinstance(item, dict):
                        interaction_type = item.get('type', 'view')
                        product_name = item.get('product', 'Unknown')
                        category = item.get('category', '')
                        
                        # Add emoji for interaction type
                        type_emoji = {
                            'purchase': '🛒',
                            'cart': '🛍️',
                            'view': '👁️',
                            'rating': '⭐'
                        }.get(interaction_type, '•')
                        
                        history_items.append(
                            f"{type_emoji} {product_name} ({interaction_type}) - {category}"
                        )
                
                if history_items:
                    history_text = "\n".join(history_items)
                else:
                    history_text = "User is new or has limited interaction history"
        
        # Build recommendation score context
        score_context = ""
        if recommendation_score is not None:
            if recommendation_score > 0.7:
                score_context = "This is a highly recommended product (high confidence score)."
            elif recommendation_score > 0.4:
                score_context = "This product has a good recommendation score."
            else:
                score_context = "This product is recommended based on similar user patterns."
        
        # Enhanced context prompt
        context = f"""You are an expert e-commerce recommendation assistant. Your task is to explain why a specific product is being recommended to a user in a personalized, engaging, and helpful manner.

USER'S SHOPPING BEHAVIOR AND PREFERENCES:
{history_text if history_text else "This is a new user with no previous interaction history."}

{score_context}

PRODUCT BEING RECOMMENDED:
- Product Name: {product_info.get('name', 'Unknown Product')}
- Category: {product_info.get('category', 'General')}
- Description: {product_info.get('description', '')[:300]}
- Tags/Features: {', '.join(product_info.get('tags', [])[:10])}

TASK:
Write a personalized explanation (2-3 sentences, maximum 150 words) that:
1. References the user's specific shopping behavior and interests
2. Explains why this product matches their preferences
3. Highlights key product features that align with their interests
4. Uses a friendly, conversational tone
5. Makes the recommendation feel personalized and relevant

Be specific about what in their history led to this recommendation. If they're a new user, explain general benefits and why similar customers like this product."""

        return context
    
    def _generate_groq_with_retry(self, context, retry_count=0):
        """Generate explanation using Groq API with retry logic"""
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Enhanced system prompt
        system_prompt = """You are a helpful and friendly e-commerce recommendation assistant. 
Your role is to explain product recommendations in a clear, personalized, and engaging way.
- Keep explanations concise (2-3 sentences, max 150 words)
- Reference specific user behaviors when possible
- Highlight product features that match user interests
- Use a conversational, friendly tone
- Make recommendations feel personalized and relevant
- If user is new, explain why similar customers like this product"""
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            "model": self.model,
            "temperature": 0.8,  # Slightly higher for more natural responses
            "max_tokens": 250,  # Increased for better explanations
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.groq_url, 
                json=data, 
                headers=headers, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                explanation = result['choices'][0]['message']['content'].strip()
                
                # Clean up explanation
                explanation = self._clean_explanation(explanation)
                return explanation
                
            elif response.status_code == 429:  # Rate limit
                if retry_count < self.max_retries:
                    wait_time = (2 ** retry_count) + 1  # Exponential backoff
                    print(f"Rate limited. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    return self._generate_groq_with_retry(context, retry_count + 1)
                else:
                    print("Max retries reached for rate limit")
                    return self._generate_fallback_explanation(context)
            
            elif response.status_code == 404:  # Model not found
                error_data = response.json() if response.text else {}
                error_msg = error_data.get('error', {}).get('message', 'Model not found')
                print(f"Groq API Error 404: {error_msg}")
                print(f"Current model: {self.model}")
                print(f"Valid models: llama-3.1-70b-versatile, llama-3.1-8b-instant, llama-3.2-3b-instant, mixtral-8x7b-32768, gemma2-9b-it")
                
                # Try with a fallback model only once
                if retry_count == 0:
                    fallback_model = 'llama-3.1-8b-instant'
                    if self.model != fallback_model:
                        print(f"Attempting to use fallback model: {fallback_model}")
                        original_model = self.model
                        self.model = fallback_model
                        try:
                            result = self._generate_groq_with_retry(context, retry_count + 1)
                            self.model = original_model  # Restore original model
                            return result
                        except:
                            self.model = original_model  # Restore original model
                            return self._generate_fallback_explanation(context)
                
                return self._generate_fallback_explanation(context)
                    
            else:
                print(f"Groq API Error: {response.status_code} - {response.text}")
                if retry_count < self.max_retries and response.status_code >= 500:
                    # Retry on server errors
                    time.sleep(1)
                    return self._generate_groq_with_retry(context, retry_count + 1)
                return self._generate_fallback_explanation(context)
                
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                print(f"Request timeout. Retrying... ({retry_count + 1}/{self.max_retries})")
                time.sleep(2)
                return self._generate_groq_with_retry(context, retry_count + 1)
            print("Groq API request timed out after retries")
            return self._generate_fallback_explanation(context)
            
        except requests.exceptions.RequestException as e:
            print(f"Groq API request failed: {e}")
            if retry_count < self.max_retries:
                time.sleep(1)
                return self._generate_groq_with_retry(context, retry_count + 1)
            return self._generate_fallback_explanation(context)
            
        except Exception as e:
            print(f"Unexpected error in Groq API call: {e}")
            return self._generate_fallback_explanation(context)
    
    def _clean_explanation(self, explanation):
        """Clean and format the explanation"""
        # Remove any markdown formatting
        explanation = explanation.replace('**', '').replace('*', '')
        explanation = explanation.replace('##', '').replace('#', '')
        
        # Remove quotes if the entire explanation is quoted
        if explanation.startswith('"') and explanation.endswith('"'):
            explanation = explanation[1:-1]
        if explanation.startswith("'") and explanation.endswith("'"):
            explanation = explanation[1:-1]
        
        # Ensure it ends with proper punctuation
        if explanation and not explanation[-1] in '.!?':
            explanation += '.'
        
        return explanation.strip()
    
    def _generate_fallback_explanation(self, context):
        """Generate fallback explanation if Groq API fails"""
        # Extract information from context
        category = "products"
        if "Category:" in context:
            category = context.split("Category:")[1].split("\n")[0].strip()
        
        product_name = "this product"
        if "Product Name:" in context:
            product_name = context.split("Product Name:")[1].split("\n")[0].strip()
        
        # Check if user has history
        has_history = "User is new" not in context and "limited interaction" not in context
        
        if has_history:
            return f"Based on your recent browsing and purchase history in the {category} category, we recommend {product_name} as it aligns with your interests and preferences. This product is popular among customers with similar shopping patterns."
        else:
            return f"We recommend {product_name} because it's a popular choice in the {category} category and matches the preferences of customers similar to you. This product offers great value and quality."

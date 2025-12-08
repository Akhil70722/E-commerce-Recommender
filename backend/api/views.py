"""
API Views for E-commerce Product Recommender.

This module contains class-based views following Python and Django best practices.
Uses CSV files for data storage instead of database.
"""
import json
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core.cache import cache
from .csv_loader import CSVDataLoader
from .engine import RecommendationEngine
from .llm_service import LLMService


class BaseAPIView(View):
    """Base class for API views with common functionality."""
    
    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        """Override dispatch to add CSRF exemption."""
        return super().dispatch(*args, **kwargs)
    
    def parse_json_body(self, request):
        """Parse JSON request body."""
        try:
            return json.loads(request.body)
        except json.JSONDecodeError:
            return None
    
    def json_response(self, data, status=200):
        """Return JSON response."""
        return JsonResponse(data, status=status, safe=False)
    
    def error_response(self, message, status=400):
        """Return error JSON response."""
        return self.json_response({'error': message}, status=status)
    
    def _get_csv_loader(self):
        """Get CSV data loader instance."""
        return CSVDataLoader()


class CategoryListView(BaseAPIView):
    """View for listing categories."""
    
    def get(self, request):
        """Get all categories."""
        loader = self._get_csv_loader()
        categories = loader.load_categories()
        data = [{
            'id': cat['id'],
            'name': cat['name'],
            'description': cat.get('description', '')
        } for cat in categories]
        return self.json_response({'categories': data})


class CategoryDetailView(BaseAPIView):
    """View for retrieving a category."""
    
    def get(self, request, pk):
        """Get a specific category."""
        loader = self._get_csv_loader()
        categories = loader.load_categories()
        category = next((c for c in categories if c['id'] == int(pk)), None)
        
        if not category:
            return self.error_response('Category not found', 404)
        
        return self.json_response({
            'id': category['id'],
            'name': category['name'],
            'description': category.get('description', '')
        })


class ProductListView(BaseAPIView):
    """View for listing products."""
    
    def get(self, request):
        """Get all products with optional filtering."""
        loader = self._get_csv_loader()
        products = loader.load_products()
        interactions = loader.load_interactions()
        
        # Filter by category
        category_id = request.GET.get('category')
        if category_id:
            products = [p for p in products if p['category']['id'] == int(category_id)]
        
        # Search functionality
        search = request.GET.get('search', '').lower()
        if search:
            products = [
                p for p in products
                if search in p['name'].lower() or
                search in p.get('description', '').lower() or
                search in p.get('tags', '').lower()
            ]
        
        # Calculate ratings for each product
        ratings_dict = {}
        for interaction in interactions:
            if interaction.get('interaction_type') == 'rating' and interaction.get('rating'):
                product_id = interaction['product_id']
                if product_id not in ratings_dict:
                    ratings_dict[product_id] = {'ratings': [], 'count': 0}
                ratings_dict[product_id]['ratings'].append(interaction['rating'])
                ratings_dict[product_id]['count'] += 1
        
        # Calculate averages
        for product_id, rating_data in ratings_dict.items():
            ratings_dict[product_id] = {
                'average_rating': round(sum(rating_data['ratings']) / len(rating_data['ratings']), 1),
                'rating_count': rating_data['count']
            }
        
        data = []
        for prod in products:
            tags = prod.get('tags', '')
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
            
            rating_info = ratings_dict.get(prod['id'], {'average_rating': 0, 'rating_count': 0})
            
            data.append({
                'id': prod['id'],
                'name': prod['name'],
                'description': prod.get('description', ''),
                'category': prod['category'],
                'price': prod['price'],
                'tags': tags,
                'tags_list': tags_list,
                'image_url': prod.get('image_url', ''),
                'stock': prod.get('stock', 0),
                'average_rating': rating_info['average_rating'],
                'rating_count': rating_info['rating_count']
            })
        
        return self.json_response({'products': data})


class ProductDetailView(BaseAPIView):
    """View for retrieving a product."""
    
    def get(self, request, pk):
        """Get a specific product."""
        loader = self._get_csv_loader()
        products = loader.load_products()
        product = next((p for p in products if p['id'] == int(pk)), None)
        
        if not product:
            return self.error_response('Product not found', 404)
        
        tags = product.get('tags', '')
        tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        
        return self.json_response({
            'id': product['id'],
            'name': product['name'],
            'description': product.get('description', ''),
            'category': product['category'],
            'price': product['price'],
            'tags': tags,
            'tags_list': tags_list,
            'image_url': product.get('image_url', ''),
            'stock': product.get('stock', 0)
        })


class UserListView(BaseAPIView):
    """View for listing users."""
    
    def get(self, request):
        """Get all users."""
        loader = self._get_csv_loader()
        users = loader.load_users()
        data = [{
            'id': user['id'],
            'name': user['name'],
            'email': user['email']
        } for user in users]
        return self.json_response({'users': data})


class UserDetailView(BaseAPIView):
    """View for retrieving a user."""
    
    def get(self, request, pk):
        """Get a specific user."""
        loader = self._get_csv_loader()
        users = loader.load_users()
        user = next((u for u in users if u['id'] == int(pk)), None)
        
        if not user:
            return self.error_response('User not found', 404)
        
        return self.json_response({
            'id': user['id'],
            'name': user['name'],
            'email': user['email']
        })


class InteractionListView(BaseAPIView):
    """View for listing user interactions."""
    
    def get(self, request):
        """Get all interactions with optional filtering."""
        loader = self._get_csv_loader()
        interactions = loader.load_interactions()
        users = {u['id']: u for u in loader.load_users()}
        products = {p['id']: p for p in loader.load_products()}
        
        # Filter by user_id
        user_id = request.GET.get('user_id')
        if user_id:
            interactions = [i for i in interactions if i['user_id'] == int(user_id)]
        
        # Filter by product_id
        product_id = request.GET.get('product_id')
        if product_id:
            interactions = [i for i in interactions if i['product_id'] == int(product_id)]
        
        data = []
        for inter in interactions:
            user = users.get(inter['user_id'], {})
            product = products.get(inter['product_id'], {})
            
            data.append({
                'id': inter.get('id', 0),
                'user': {
                    'id': user.get('id', 0),
                    'name': user.get('name', ''),
                    'email': user.get('email', '')
                },
                'product': {
                    'id': product.get('id', 0),
                    'name': product.get('name', '')
                },
                'interaction_type': inter.get('interaction_type', ''),
                'rating': inter.get('rating'),
                'timestamp': inter.get('timestamp', '')
            })
        
        return self.json_response({'interactions': data})


class RecommendationListView(BaseAPIView):
    """View for getting personalized recommendations with LLM explanations."""
    
    def __init__(self, *args, **kwargs):
        """Initialize recommendation engine and LLM service."""
        super().__init__(*args, **kwargs)
        # Pass Django cache to engine
        self.engine = RecommendationEngine(cache_handler=cache)
        try:
            self.llm_service = LLMService()
        except ValueError as e:
            print(f"Warning: {e}")
            self.llm_service = None
    
    def _get_all_data(self):
        """Load all data from CSV files."""
        loader = self._get_csv_loader()
        return {
            'interactions_data': loader.load_interactions(),
            'browsing_data': loader.load_browsing_history(),
            'wishlist_data': loader.load_wishlist(),
            'search_history_data': loader.load_search_history(),
            'products_data': loader.load_products(),
            'users_data': loader.load_users()
        }
    
    def get(self, request, user_id):
        """Get real-time recommendations for a specific user."""
        user_id = int(user_id)
        loader = self._get_csv_loader()
        users = loader.load_users()
        user = next((u for u in users if u['id'] == user_id), None)
        
        if not user:
            return self.error_response('User not found', 404)
        
        # Get real-time recommendations (always uses latest data)
        real_time = request.GET.get('real_time', 'true').lower() == 'true'
        
        # Load all data from CSV
        data = self._get_all_data()
        
        try:
            # Dynamic recommendation count
            n_recommendations = int(request.GET.get('limit', 20))
            recommendations_data = self.engine.hybrid_recommendation(
                user_id,
                data['interactions_data'],
                data['browsing_data'],
                data['wishlist_data'],
                data['search_history_data'],
                data['products_data'],
                data['users_data'],
                n_recommendations=n_recommendations,
                real_time=real_time
            )
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            return self.error_response(
                f"Error generating recommendations: {str(e)}",
                500
            )
        
        if not recommendations_data:
            return self.json_response({
                'user_id': user_id,
                'message': 'No recommendations available. Please interact with more products to get personalized recommendations.',
                'recommendations': []
            })
        
        # Get user's interaction history for context
        products = {p['id']: p for p in data['products_data']}
        interactions = data['interactions_data']
        browsing = data['browsing_data']
        
        # Get user interactions
        user_interactions = [
            i for i in interactions 
            if i.get('user_id') == user_id
        ]
        user_browsing = [
            b for b in browsing 
            if b.get('user_id') == user_id
        ]
        
        # Build user history
        user_history = []
        for interaction in sorted(user_interactions, key=lambda x: x.get('timestamp', ''), reverse=True)[:15]:
            product = products.get(interaction.get('product_id'))
            if product:
                user_history.append({
                    'product': product.get('name', ''),
                    'type': interaction.get('interaction_type', ''),
                    'category': product.get('category', {}).get('name', ''),
                    'timestamp': interaction.get('timestamp', '')
                })
        
        for browse in sorted(user_browsing, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]:
            product = products.get(browse.get('product_id'))
            if product:
                user_history.append({
                    'product': product.get('name', ''),
                    'type': 'browse',
                    'category': product.get('category', {}).get('name', ''),
                    'timestamp': browse.get('timestamp', '')
                })
        
        # Sort by timestamp
        user_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        user_history = user_history[:15]
        
        # Generate recommendations with LLM explanations
        recommendations = []
        for product_id, score in recommendations_data:
            product = products.get(product_id)
            if not product:
                continue
            
            # Get product info
            tags = product.get('tags', '')
            tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
            
            product_info = {
                'name': product.get('name', ''),
                'category': product.get('category', {}).get('name', ''),
                'description': product.get('description', ''),
                'tags': tags_list
            }
            
            # Generate explanation
            explanation = None
            if self.llm_service:
                try:
                    explanation = self.llm_service.generate_explanation(
                        user_id=user_id,
                        product_id=product_id,
                        user_history=user_history,
                        product_info=product_info,
                        recommendation_score=score
                    )
                except Exception as e:
                    print(f"Error generating LLM explanation: {e}")
                    explanation = self._generate_default_explanation(product, user_history, score)
            else:
                explanation = self._generate_default_explanation(product, user_history, score)
            
            # Calculate average rating
            product_ratings = [
                i.get('rating') for i in interactions
                if i.get('product_id') == product_id and
                i.get('interaction_type') == 'rating' and
                i.get('rating') is not None
            ]
            
            avg_rating = round(sum(product_ratings) / len(product_ratings), 1) if product_ratings else 0
            rating_count = len(product_ratings)
            
            recommendations.append({
                'product_id': product['id'],
                'product_name': product.get('name', ''),
                'category': product.get('category', {}).get('name', ''),
                'price': product.get('price', 0),
                'description': product.get('description', ''),
                'image_url': product.get('image_url', ''),
                'score': round(score, 3),
                'explanation': explanation,
                'average_rating': avg_rating,
                'rating_count': rating_count,
                'stock': product.get('stock', 0),
                'tags': tags_list
            })
        
        return self.json_response({
            'user_id': user_id,
            'user_email': user.get('email', ''),
            'real_time': real_time,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        })
    
    def _generate_default_explanation(self, product, user_history, score=None):
        """Generate a default explanation when LLM is not available"""
        category_name = product.get('category', {}).get('name', 'product')
        
        if user_history:
            categories = [h.get('category', '') for h in user_history if isinstance(h, dict)]
            if category_name in categories:
                return f"Based on your recent interest in {category_name} products, we recommend this item as it matches your preferences and shopping patterns."
            
            product_names = [h.get('product', '') for h in user_history if isinstance(h, dict)]
            if any(product.get('name', '').lower() in name.lower() or name.lower() in product.get('name', '').lower() 
                   for name in product_names if name):
                return f"Recommended because you've shown interest in similar products. This {category_name} item complements your recent selections."
            
            return f"Recommended based on your browsing and purchase history. This product aligns with your shopping preferences."
        
        if score and score > 0.6:
            return f"This {category_name} product is highly recommended because it's popular among customers with similar interests and offers great value."
        
        return f"This product matches your preferences and is popular with similar customers. We think you'll enjoy this {category_name} item."


class BrowsingHistoryView(BaseAPIView):
    """View for tracking browsing history (read-only from CSV)."""
    
    def get(self, request):
        """Get browsing history."""
        loader = self._get_csv_loader()
        browsing = loader.load_browsing_history()
        
        user_id = request.GET.get('user_id')
        if user_id:
            browsing = [b for b in browsing if b.get('user_id') == int(user_id)]
        
        return self.json_response({'browsing_history': browsing})


class SearchHistoryView(BaseAPIView):
    """View for tracking search queries (read-only from CSV)."""
    
    def get(self, request):
        """Get search history."""
        loader = self._get_csv_loader()
        searches = loader.load_search_history()
        
        user_id = request.GET.get('user_id')
        if user_id:
            searches = [s for s in searches if s.get('user_id') == int(user_id)]
        
        return self.json_response({'search_history': searches})


class WishlistView(BaseAPIView):
    """View for managing wishlist items."""
    
    def get(self, request, user_id):
        """Get user's wishlist."""
        loader = self._get_csv_loader()
        wishlist = loader.load_wishlist()
        products = {p['id']: p for p in loader.load_products()}
        
        user_id = int(user_id)
        wishlist_items = [w for w in wishlist if w.get('user_id') == user_id]
        
        items = []
        for item in wishlist_items:
            product = products.get(item.get('product_id'))
            if product:
                items.append({
                    'id': item.get('id', 0),
                    'product_id': product['id'],
                    'product': {
                        'id': product['id'],
                        'name': product.get('name', ''),
                        'category': product.get('category', {}).get('name', ''),
                        'price': product.get('price', 0),
                        'description': product.get('description', ''),
                        'image_url': product.get('image_url', ''),
                    }
                })
        
        return self.json_response({'items': items})


class ModelTrainingView(BaseAPIView):
    """View for training the recommendation model."""
    
    def __init__(self, *args, **kwargs):
        """Initialize recommendation engine."""
        super().__init__(*args, **kwargs)
        # Pass Django cache to engine
        self.engine = RecommendationEngine(cache_handler=cache)
    
    def _get_all_data(self):
        """Load all data from CSV files."""
        loader = self._get_csv_loader()
        return {
            'interactions_data': loader.load_interactions(),
            'browsing_data': loader.load_browsing_history(),
            'wishlist_data': loader.load_wishlist(),
            'products_data': loader.load_products()
        }
    
    def post(self, request):
        """Train the recommendation model."""
        force_retrain = request.GET.get('force', 'false').lower() == 'true'
        
        # Load all data from CSV
        data = self._get_all_data()
        
        try:
            success = self.engine.train_model(
                data['interactions_data'],
                data['browsing_data'],
                data['wishlist_data'],
                data['products_data'],
                force_retrain=force_retrain
            )
            
            if success:
                return self.json_response({
                    'message': 'Model training completed successfully',
                    'trained_at': self.engine.last_training_time.isoformat() if self.engine.last_training_time else None,
                    'force_retrain': force_retrain
                })
            else:
                return self.json_response({
                    'message': 'Model training skipped (recently trained or insufficient data)',
                    'force_retrain': force_retrain
                }, status=200)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self.error_response(
                f'Error training model: {str(e)}',
                500
            )
    
    def get(self, request):
        """Get model training status."""
        last_training = cache.get('last_model_training')
        
        return self.json_response({
            'last_training': last_training.isoformat() if last_training else None,
            'model_cached': cache.get('recommendation_model_cache') is not None,
            'vectorizer_cached': cache.get('content_vectorizer') is not None
        })

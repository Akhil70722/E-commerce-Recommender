"""
API Views for E-commerce Product Recommender.

This module contains class-based views following Python and Django best practices.
"""
import json
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.shortcuts import get_object_or_404
from django.db.models import Q, Count, Avg
from django.db import models
from django.core.cache import cache
from .models import (
    Category, Product, User, UserInteraction, Recommendation,
    BrowsingHistory, SearchHistory, Wishlist
)
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


class CategoryListView(BaseAPIView):
    """View for listing and creating categories."""
    
    def get(self, request):
        """Get all categories."""
        categories = Category.objects.all()
        data = [{
            'id': cat.id,
            'name': cat.name,
            'description': cat.description,
            'created_at': cat.created_at.isoformat()
        } for cat in categories]
        return self.json_response({'categories': data})
    
    def post(self, request):
        """Create a new category."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        category = Category.objects.create(
            name=data.get('name'),
            description=data.get('description', '')
        )
        return self.json_response({
            'id': category.id,
            'name': category.name,
            'description': category.description,
            'created_at': category.created_at.isoformat()
        }, status=201)


class CategoryDetailView(BaseAPIView):
    """View for retrieving, updating, and deleting a category."""
    
    def get(self, request, pk):
        """Get a specific category."""
        category = get_object_or_404(Category, pk=pk)
        return self.json_response({
            'id': category.id,
            'name': category.name,
            'description': category.description,
            'created_at': category.created_at.isoformat()
        })
    
    def put(self, request, pk):
        """Update a category."""
        category = get_object_or_404(Category, pk=pk)
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        category.name = data.get('name', category.name)
        category.description = data.get('description', category.description)
        category.save()
        
        return self.json_response({
            'id': category.id,
            'name': category.name,
            'description': category.description,
            'created_at': category.created_at.isoformat()
        })
    
    def delete(self, request, pk):
        """Delete a category."""
        category = get_object_or_404(Category, pk=pk)
        category.delete()
        return self.json_response({'message': 'Category deleted'}, status=204)


class ProductListView(BaseAPIView):
    """View for listing and creating products."""
    
    def get(self, request):
        """Get all products with optional filtering."""
        products = Product.objects.select_related('category').all()
        
        # Filter by category
        category_id = request.GET.get('category')
        if category_id:
            products = products.filter(category_id=category_id)
        
        # Search functionality
        search = request.GET.get('search')
        if search:
            products = products.filter(
                Q(name__icontains=search) |
                Q(description__icontains=search) |
                Q(tags__icontains=search)
            )
        
        # Calculate ratings for each product dynamically
        product_ids = [prod.id for prod in products]
        ratings_data = UserInteraction.objects.filter(
            product_id__in=product_ids,
            interaction_type='rating',
            rating__isnull=False
        ).values('product_id').annotate(
            avg_rating=Avg('rating'),
            rating_count=Count('id')
        )
        
        ratings_dict = {
            item['product_id']: {
                'average_rating': round(item['avg_rating'], 1) if item['avg_rating'] else 0,
                'rating_count': item['rating_count']
            }
            for item in ratings_data
        }
        
        data = [{
            'id': prod.id,
            'name': prod.name,
            'description': prod.description,
            'category': {
                'id': prod.category.id,
                'name': prod.category.name
            },
            'price': float(prod.price),
            'tags': prod.tags,
            'tags_list': prod.get_tags_list(),
            'image_url': prod.image_url,
            'stock': prod.stock,
            'average_rating': ratings_dict.get(prod.id, {}).get('average_rating', 0),
            'rating_count': ratings_dict.get(prod.id, {}).get('rating_count', 0),
            'created_at': prod.created_at.isoformat(),
            'updated_at': prod.updated_at.isoformat()
        } for prod in products]
        
        return self.json_response({'products': data})
    
    def post(self, request):
        """Create a new product."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        product = Product.objects.create(
            name=data.get('name'),
            description=data.get('description'),
            category_id=data.get('category_id'),
            price=data.get('price'),
            tags=data.get('tags', ''),
            image_url=data.get('image_url', ''),
            stock=data.get('stock', 0)
        )
        
        return self.json_response({
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'category': {
                'id': product.category.id,
                'name': product.category.name
            },
            'price': float(product.price),
            'tags': product.tags,
            'stock': product.stock,
            'created_at': product.created_at.isoformat()
        }, status=201)


class ProductDetailView(BaseAPIView):
    """View for retrieving, updating, and deleting a product."""
    
    def get(self, request, pk):
        """Get a specific product."""
        product = get_object_or_404(
            Product.objects.select_related('category'), 
            pk=pk
        )
        return self.json_response({
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'category': {
                'id': product.category.id,
                'name': product.category.name
            },
            'price': float(product.price),
            'tags': product.tags,
            'tags_list': product.get_tags_list(),
            'image_url': product.image_url,
            'stock': product.stock,
            'created_at': product.created_at.isoformat(),
            'updated_at': product.updated_at.isoformat()
        })
    
    def put(self, request, pk):
        """Update a product."""
        product = get_object_or_404(Product, pk=pk)
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        # Update fields if provided
        if 'name' in data:
            product.name = data['name']
        if 'description' in data:
            product.description = data['description']
        if 'category_id' in data:
            product.category_id = data['category_id']
        if 'price' in data:
            product.price = data['price']
        if 'tags' in data:
            product.tags = data['tags']
        if 'stock' in data:
            product.stock = data['stock']
        
        product.save()
        
        return self.json_response({
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'category': {
                'id': product.category.id,
                'name': product.category.name
            },
            'price': float(product.price),
            'tags': product.tags,
            'stock': product.stock
        })
    
    def delete(self, request, pk):
        """Delete a product."""
        product = get_object_or_404(Product, pk=pk)
        product.delete()
        return self.json_response({'message': 'Product deleted'}, status=204)


class UserListView(BaseAPIView):
    """View for listing and creating users."""
    
    def get(self, request):
        """Get all users."""
        users = User.objects.all()
        data = [{
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        } for user in users]
        return self.json_response({'users': data})
    
    def post(self, request):
        """Create a new user."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        user = User.objects.create(
            name=data.get('name'),
            email=data.get('email')
        )
        
        return self.json_response({
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        }, status=201)


class UserDetailView(BaseAPIView):
    """View for retrieving, updating, and deleting a user."""
    
    def get(self, request, pk):
        """Get a specific user."""
        user = get_object_or_404(User, pk=pk)
        return self.json_response({
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        })
    
    def put(self, request, pk):
        """Update a user."""
        user = get_object_or_404(User, pk=pk)
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        if 'name' in data:
            user.name = data['name']
        if 'email' in data:
            user.email = data['email']
        
        user.save()
        
        return self.json_response({
            'id': user.id,
            'name': user.name,
            'email': user.email,
            'created_at': user.created_at.isoformat()
        })
    
    def delete(self, request, pk):
        """Delete a user."""
        user = get_object_or_404(User, pk=pk)
        user.delete()
        return self.json_response({'message': 'User deleted'}, status=204)


class InteractionListView(BaseAPIView):
    """View for listing and creating user interactions."""
    
    def get(self, request):
        """Get all interactions with optional filtering."""
        interactions = UserInteraction.objects.select_related('user', 'product').all()
        
        # Filter by user_id
        user_id = request.GET.get('user_id')
        if user_id:
            interactions = interactions.filter(user_id=user_id)
        
        # Filter by product_id
        product_id = request.GET.get('product_id')
        if product_id:
            interactions = interactions.filter(product_id=product_id)
        
        data = [{
            'id': inter.id,
            'user': {
                'id': inter.user.id,
                'name': inter.user.name,
                'email': inter.user.email
            },
            'product': {
                'id': inter.product.id,
                'name': inter.product.name
            },
            'interaction_type': inter.interaction_type,
            'rating': inter.rating,
            'timestamp': inter.timestamp.isoformat()
        } for inter in interactions]
        
        return self.json_response({'interactions': data})
    
    def post(self, request):
        """Create a new interaction."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        interaction = UserInteraction.objects.create(
            user_id=data.get('user_id'),
            product_id=data.get('product_id'),
            interaction_type=data.get('interaction_type'),
            rating=data.get('rating')
        )
        
        return self.json_response({
            'id': interaction.id,
            'user_id': interaction.user.id,
            'product_id': interaction.product.id,
            'interaction_type': interaction.interaction_type,
            'rating': interaction.rating,
            'timestamp': interaction.timestamp.isoformat()
        }, status=201)


class RecommendationListView(BaseAPIView):
    """View for getting personalized recommendations with LLM explanations."""
    
    def __init__(self, *args, **kwargs):
        """Initialize recommendation engine and LLM service."""
        super().__init__(*args, **kwargs)
        self.engine = RecommendationEngine()
        try:
            self.llm_service = LLMService()
        except ValueError as e:
            print(f"Warning: {e}")
            self.llm_service = None
    
    def get(self, request, user_id):
        """Get real-time recommendations for a specific user."""
        user = get_object_or_404(User, pk=user_id)
        
        # Get real-time recommendations (always uses latest data)
        real_time = request.GET.get('real_time', 'true').lower() == 'true'
        
        try:
            # Dynamic recommendation count - can be configured via query param or settings
            n_recommendations = int(request.GET.get('limit', 20))
            recommendations_data = self.engine.hybrid_recommendation(
                user_id, 
                n_recommendations=n_recommendations,
                real_time=real_time
            )
        except Exception as e:
            print(f"Error generating recommendations: {e}")
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
        
        # Get comprehensive user's interaction history for better context
        # Dynamic limits - can be configured via query params
        interaction_limit = int(request.GET.get('interaction_limit', 15))
        browsing_limit = int(request.GET.get('browsing_limit', 10))
        
        user_interactions = UserInteraction.objects.filter(
            user=user
        ).select_related('product', 'product__category').order_by('-timestamp')[:interaction_limit]
        
        user_browsing = BrowsingHistory.objects.filter(
            user=user
        ).select_related('product', 'product__category').order_by('-timestamp')[:browsing_limit]
        
        user_history = []
        for interaction in user_interactions:
            user_history.append({
                'product': interaction.product.name,
                'type': interaction.interaction_type,
                'category': interaction.product.category.name,
                'timestamp': interaction.timestamp.isoformat()
            })
        
        for browsing in user_browsing:
            user_history.append({
                'product': browsing.product.name,
                'type': 'browse',
                'category': browsing.product.category.name,
                'timestamp': browsing.timestamp.isoformat()
            })
        
        # Sort by timestamp (most recent first)
        user_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        user_history = user_history[:15]  # Top 15 most recent
        
        # Generate recommendations with enhanced LLM explanations
        recommendations = []
        for product_id, score in recommendations_data:
            try:
                product = Product.objects.select_related('category').get(id=product_id)
            except Product.DoesNotExist:
                continue
            
            # Get product info
            product_info = {
                'name': product.name,
                'category': product.category.name,
                'description': product.description,
                'tags': product.get_tags_list()
            }
            
            # Generate explanation (with LLM if available, otherwise use default)
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
            
            # Create or update recommendation
            Recommendation.objects.update_or_create(
                user=user,
                product=product,
                defaults={
                    'score': score,
                    'explanation': explanation
                }
            )
            
            # Calculate average rating dynamically
            ratings = UserInteraction.objects.filter(
                product=product,
                interaction_type='rating',
                rating__isnull=False
            )
            avg_rating = ratings.aggregate(Avg('rating'))['rating__avg'] or 0
            rating_count = ratings.count()
            
            recommendations.append({
                'product_id': product.id,
                'product_name': product.name,
                'category': product.category.name,
                'price': float(product.price),
                'description': product.description,
                'image_url': product.image_url,
                'score': round(score, 3),
                'explanation': explanation,
                'average_rating': round(avg_rating, 1) if avg_rating else 0,
                'rating_count': rating_count,
                'stock': product.stock,
                'tags': product.get_tags_list()
            })
        
        return self.json_response({
            'user_id': user_id,
            'user_email': user.email,
            'real_time': real_time,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        })
    
    def _generate_default_explanation(self, product, user_history, score=None):
        """Generate a default explanation when LLM is not available"""
        if user_history:
            categories = [h.get('category', '') for h in user_history if isinstance(h, dict)]
            if product.category.name in categories:
                return f"Based on your recent interest in {product.category.name} products, we recommend this item as it matches your preferences and shopping patterns."
            
            # Check for specific product interactions
            product_names = [h.get('product', '') for h in user_history if isinstance(h, dict)]
            if any(product.name.lower() in name.lower() or name.lower() in product.name.lower() 
                   for name in product_names if name):
                return f"Recommended because you've shown interest in similar products. This {product.category.name} item complements your recent selections."
            
            return f"Recommended based on your browsing and purchase history. This product aligns with your shopping preferences."
        
        if score and score > 0.6:
            return f"This {product.category.name} product is highly recommended because it's popular among customers with similar interests and offers great value."
        
        return f"This product matches your preferences and is popular with similar customers. We think you'll enjoy this {product.category.name} item."


class BrowsingHistoryView(BaseAPIView):
    """View for tracking browsing history."""
    
    def post(self, request):
        """Record a browsing event."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        browsing = BrowsingHistory.objects.create(
            user_id=data.get('user_id'),
            product_id=data.get('product_id'),
            time_spent=data.get('time_spent', 0)
        )
        
        # Also create a view interaction
        UserInteraction.objects.get_or_create(
            user_id=data.get('user_id'),
            product_id=data.get('product_id'),
            interaction_type='view',
            defaults={'rating': None}
        )
        
        return self.json_response({
            'id': browsing.id,
            'user_id': browsing.user.id,
            'product_id': browsing.product.id,
            'timestamp': browsing.timestamp.isoformat()
        }, status=201)


class SearchHistoryView(BaseAPIView):
    """View for tracking search queries."""
    
    def post(self, request):
        """Record a search query."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        # Count search results
        query = data.get('query', '')
        from django.db.models import Q
        results_count = Product.objects.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(tags__icontains=query)
        ).count()
        
        search = SearchHistory.objects.create(
            user_id=data.get('user_id'),
            query=query,
            results_count=results_count
        )
        
        return self.json_response({
            'id': search.id,
            'user_id': search.user.id,
            'query': search.query,
            'results_count': search.results_count,
            'timestamp': search.timestamp.isoformat()
        }, status=201)


class WishlistView(BaseAPIView):
    """View for managing wishlist items."""
    
    def post(self, request):
        """Add item to wishlist."""
        data = self.parse_json_body(request)
        if not data:
            return self.error_response('Invalid JSON', 400)
        
        wishlist_item, created = Wishlist.objects.get_or_create(
            user_id=data.get('user_id'),
            product_id=data.get('product_id')
        )
        
        if not created:
            return self.json_response({
                'message': 'Item already in wishlist',
                'id': wishlist_item.id
            })
        
        return self.json_response({
            'id': wishlist_item.id,
            'user_id': wishlist_item.user.id,
            'product_id': wishlist_item.product.id,
            'added_at': wishlist_item.added_at.isoformat()
        }, status=201)
    
    def get(self, request, user_id):
        """Get user's wishlist."""
        wishlist_items = Wishlist.objects.filter(
            user_id=user_id
        ).select_related('product', 'product__category')
        
        items = [{
            'id': item.id,
            'product_id': item.product.id,
            'product': {
                'id': item.product.id,
                'name': item.product.name,
                'category': item.product.category.name,
                'price': float(item.product.price),
                'description': item.product.description,
                'image_url': item.product.image_url,
            },
            'added_at': item.added_at.isoformat()
        } for item in wishlist_items]
        
        return self.json_response({'items': items})
    
    def delete(self, request, user_id, product_id):
        """Remove item from wishlist."""
        wishlist_item = get_object_or_404(
            Wishlist,
            user_id=user_id,
            product_id=product_id
        )
        wishlist_item.delete()
        return self.json_response({'message': 'Item removed from wishlist'}, status=204)


class ModelTrainingView(BaseAPIView):
    """View for training the recommendation model."""
    
    def __init__(self, *args, **kwargs):
        """Initialize recommendation engine."""
        super().__init__(*args, **kwargs)
        self.engine = RecommendationEngine()
    
    def post(self, request):
        """Train the recommendation model."""
        force_retrain = request.GET.get('force', 'false').lower() == 'true'
        
        try:
            success = self.engine.train_model(force_retrain=force_retrain)
            
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

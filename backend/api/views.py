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
from django.db.models import Q
from .models import Category, Product, User, UserInteraction, Recommendation
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
        """Get recommendations for a specific user."""
        user = get_object_or_404(User, pk=user_id)
        
        if not self.llm_service:
            return self.error_response(
                'LLM service not configured. Please set GROQ_API_KEY in .env file.',
                500
            )
        
        # Get recommendations using hybrid approach
        recommendations_data = self.engine.hybrid_recommendation(
            user_id, 
            n_recommendations=10
        )
        
        if not recommendations_data:
            return self.json_response({
                'user_id': user_id,
                'message': 'No recommendations available. Please interact with more products.',
                'recommendations': []
            })
        
        # Get user's interaction history for context
        user_interactions = UserInteraction.objects.filter(
            user=user
        ).select_related('product')[:10]
        
        user_history = [
            {
                'product': interaction.product.name,
                'type': interaction.interaction_type,
                'category': interaction.product.category.name
            }
            for interaction in user_interactions
        ]
        
        # Generate recommendations with Groq explanations
        recommendations = []
        for product_id, score in recommendations_data:
            product = Product.objects.select_related('category').get(id=product_id)
            
            # Get product info
            product_info = {
                'name': product.name,
                'category': product.category.name,
                'description': product.description,
                'tags': product.get_tags_list()
            }
            
            # Generate Groq LLM explanation
            explanation = self.llm_service.generate_explanation(
                user_id=user_id,
                product_id=product_id,
                user_history=user_history,
                product_info=product_info
            )
            
            # Create or update recommendation
            Recommendation.objects.update_or_create(
                user=user,
                product=product,
                defaults={
                    'score': score,
                    'explanation': explanation
                }
            )
            
            recommendations.append({
                'product_id': product.id,
                'product_name': product.name,
                'category': product.category.name,
                'price': float(product.price),
                'description': product.description,
                'image_url': product.image_url,
                'score': round(score, 3),
                'explanation': explanation
            })
        
        return self.json_response({
            'user_id': user_id,
            'user_email': user.email,
            'recommendations': recommendations
        })

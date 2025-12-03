"""
URL configuration for API endpoints.
"""
from django.urls import path
from . import views

urlpatterns = [
    # Categories
    path('categories/', views.CategoryListView.as_view(), name='category-list'),
    path('categories/<int:pk>/', views.CategoryDetailView.as_view(), name='category-detail'),
    
    # Products
    path('products/', views.ProductListView.as_view(), name='product-list'),
    path('products/<int:pk>/', views.ProductDetailView.as_view(), name='product-detail'),
    
    # Users
    path('users/', views.UserListView.as_view(), name='user-list'),
    path('users/<int:pk>/', views.UserDetailView.as_view(), name='user-detail'),
    
    # Interactions
    path('interactions/', views.InteractionListView.as_view(), name='interaction-list'),
    
    # Recommendations
    path('recommendations/user/<int:user_id>/', 
         views.RecommendationListView.as_view(), 
         name='recommendation-list'),
]

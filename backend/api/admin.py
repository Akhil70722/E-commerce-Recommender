from django.contrib import admin
from .models import Category, Product, User, UserInteraction, Recommendation


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'description', 'created_at']
    search_fields = ['name']


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'price', 'stock', 'created_at']
    list_filter = ['category', 'created_at']
    search_fields = ['name', 'description', 'tags']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'created_at']
    search_fields = ['name', 'email']
    readonly_fields = ['created_at']


@admin.register(UserInteraction)
class UserInteractionAdmin(admin.ModelAdmin):
    list_display = ['user', 'product', 'interaction_type', 'rating', 'timestamp']
    list_filter = ['interaction_type', 'timestamp']
    search_fields = ['user__email', 'product__name']
    readonly_fields = ['timestamp']


@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['user', 'product', 'score', 'created_at']
    list_filter = ['created_at', 'score']
    search_fields = ['user__email', 'product__name']
    readonly_fields = ['created_at']


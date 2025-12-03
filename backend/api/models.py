from django.db import models

# Category Model
class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']


# Product Model
class Product(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    price = models.DecimalField(max_digits=10, decimal_places=2)
    tags = models.CharField(max_length=500, blank=True, help_text="Comma-separated tags")
    image_url = models.URLField(blank=True)
    stock = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name

    def get_tags_list(self):
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]


# User Model
class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email

    class Meta:
        ordering = ['-created_at']


# Interaction Type Choices
class InteractionType(models.TextChoices):
    VIEW = 'view', 'View'
    PURCHASE = 'purchase', 'Purchase'
    CART = 'cart', 'Add to Cart'
    RATING = 'rating', 'Rating'


# User Interaction Model
class UserInteraction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='interactions')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='interactions')
    interaction_type = models.CharField(max_length=20, choices=InteractionType.choices)
    rating = models.IntegerField(null=True, blank=True, help_text="1-5 rating if interaction_type is rating")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'timestamp']),
            models.Index(fields=['product', 'timestamp']),
        ]

    def __str__(self):
        return f"{self.user.email} - {self.interaction_type} - {self.product.name}"


# Recommendation Model
class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recommendations')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='recommendations')
    score = models.FloatField(help_text="Recommendation score (0-1)")
    explanation = models.TextField(help_text="LLM-generated explanation")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-score', '-created_at']
        unique_together = [['user', 'product']]

    def __str__(self):
        return f"{self.user.email} - {self.product.name} (score: {self.score:.2f})"


import { useEffect, useState, useCallback, useMemo } from "react";
import {
  fetchUsers,
  fetchProducts,
  fetchRecommendations,
  createInteraction,
  fetchCategories,
  trackBrowsing,
  trackSearch,
  addToWishlist,
  removeFromWishlist,
  getWishlist,
} from "./api";
import "./App.css";

function App() {
  const [users, setUsers] = useState([]);
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loadingRec, setLoadingRec] = useState(false);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [wishlist, setWishlist] = useState([]);
  const [viewedProducts, setViewedProducts] = useState([]);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [cart, setCart] = useState([]);
  const [isInitializing, setIsInitializing] = useState(true);
  const [windowWidth] = useState(1200);

  const recommendationLimit = useMemo(() => {
    return Math.max(16, Math.floor(windowWidth / 200) * 4);
  }, [windowWidth]);

  const sectionLimits = useMemo(() => {
    const total = recommendations.length;
    if (total === 0) return { main: 0, crossSell: 0, frequentlyBought: 0 };
    const main = Math.min(Math.floor(total * 0.5), 8);
    const crossSell = Math.min(Math.floor(total * 0.25), 4);
    const frequentlyBought = Math.min(Math.floor(total * 0.25), 4);
    return { main, crossSell, frequentlyBought };
  }, [recommendations.length]);

  const stockThreshold = useMemo(() => {
    if (products.length === 0) return 10;
    const avgStock = products.reduce((sum, p) => sum + (p.stock || 0), 0) / products.length;
    return Math.max(5, Math.floor(avgStock * 0.2));
  }, [products]);

  const descriptionMaxLength = useMemo(() => {
    if (windowWidth < 480) return 80;
    if (windowWidth < 768) return 100;
    if (windowWidth < 1200) return 120;
    return 150;
  }, [windowWidth]);

  async function loadUserData(userId) {
    if (!userId) return;
    try {
      const wishlistData = await getWishlist(userId);
      setWishlist(Array.isArray(wishlistData) ? wishlistData : []);
    } catch (err) {
      console.error("Failed to load wishlist:", err);
    }
  }

  const handleGetRecommendations = useCallback(async () => {
    if (!selectedUserId) return;
    try {
      setLoadingRec(true);
      setError("");
      const data = await fetchRecommendations(selectedUserId, recommendationLimit);
      setRecommendations(Array.isArray(data?.recommendations) ? data.recommendations : []);
      setShowRecommendations(true);
    } catch (err) {
      console.error("Recommendation error:", err);
      setError(err.message || "Failed to fetch recommendations");
    } finally {
      setLoadingRec(false);
    }
  }, [selectedUserId, recommendationLimit]);

  useEffect(() => {
    let isMounted = true;
    let timeoutId;
    
    async function loadInitialData() {
      try {
        setIsInitializing(true);
        setError("");
        
        console.log("Loading initial data...");
        
        const [u, p, c] = await Promise.allSettled([
          fetchUsers().catch(e => { console.error("Users error:", e); throw e; }),
          fetchProducts().catch(e => { console.error("Products error:", e); throw e; }),
          fetchCategories().catch(e => { console.error("Categories error:", e); throw e; }),
        ]);
        
        if (!isMounted) return;
        
        const usersData = u.status === 'fulfilled' ? (u.value || []) : [];
        const productsData = p.status === 'fulfilled' ? (p.value || []) : [];
        const categoriesData = c.status === 'fulfilled' ? (c.value || []) : [];
        
        setUsers(Array.isArray(usersData) ? usersData : []);
        setProducts(Array.isArray(productsData) ? productsData : []);
        setCategories(Array.isArray(categoriesData) ? categoriesData : []);
        
        if (usersData.length > 0) {
          setSelectedUserId(String(usersData[0].id));
        }
        
        if (u.status === 'rejected' || p.status === 'rejected' || c.status === 'rejected') {
          const errors = [u.status === 'rejected' && u.reason?.message, 
                         p.status === 'rejected' && p.reason?.message,
                         c.status === 'rejected' && c.reason?.message].filter(Boolean);
          setError(`Connection error. Ensure backend is running at http://127.0.0.1:8000. ${errors.join('; ')}`);
        }
      } catch (err) {
        console.error("Load error:", err);
        if (isMounted) {
          setError(`Failed to load: ${err.message || 'Unknown error'}`);
        }
      } finally {
        if (isMounted) {
          setIsInitializing(false);
        }
      }
    }
    
    timeoutId = setTimeout(loadInitialData, 100);
    
    return () => {
      isMounted = false;
      clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
    if (!selectedUserId) return;
    loadUserData(selectedUserId);
    const timer = setTimeout(() => {
      handleGetRecommendations();
    }, 500);
    return () => clearTimeout(timer);
  }, [selectedUserId, handleGetRecommendations]);

  async function handleProductClick(productId) {
    if (!selectedUserId) return;
    try {
      await trackBrowsing(selectedUserId, productId);
      setViewedProducts((prev) => prev.includes(productId) ? prev : [...prev, productId]);
    } catch (err) {
      console.error("Track browsing error:", err);
    }
  }

  async function handleSearch(query) {
    setSearchQuery(query);
    if (!selectedUserId || !query.trim()) return;
    try {
      await trackSearch(selectedUserId, query);
      if (showRecommendations) {
        setTimeout(() => handleGetRecommendations(), 1000);
      }
    } catch (err) {
      console.error("Track search error:", err);
    }
  }

  async function handleAddToCart(productId) {
    if (!selectedUserId) {
      setError("Please select a user first.");
      return;
    }
    try {
      await createInteraction({
        user_id: Number(selectedUserId),
        product_id: productId,
        interaction_type: "cart",
      });
      const product = products.find((p) => p.id === productId);
      if (product) {
        setCart((prev) => [...prev, product]);
      }
      setTimeout(() => handleGetRecommendations(), 500);
    } catch (err) {
      console.error("Add to cart error:", err);
      setError("Failed to add to cart.");
    }
  }

  async function handlePurchase(productId) {
    if (!selectedUserId) {
      setError("Please select a user first.");
      return;
    }
    try {
      await createInteraction({
        user_id: Number(selectedUserId),
        product_id: productId,
        interaction_type: "purchase",
      });
      setTimeout(() => handleGetRecommendations(), 500);
    } catch (err) {
      console.error("Purchase error:", err);
      setError("Failed to record purchase.");
    }
  }

  async function handleWishlistToggle(productId) {
    if (!selectedUserId) return;
    const isInWishlist = wishlist.some((item) => item.product_id === productId);
    try {
      if (isInWishlist) {
        await removeFromWishlist(selectedUserId, productId);
        setWishlist((prev) => prev.filter((item) => item.product_id !== productId));
      } else {
        await addToWishlist(selectedUserId, productId);
        const product = products.find((p) => p.id === productId);
        if (product) {
          setWishlist((prev) => [...prev, { product_id: productId, product }]);
        }
      }
      setTimeout(() => handleGetRecommendations(), 500);
    } catch (err) {
      console.error("Wishlist error:", err);
    }
  }

  const filteredProducts = products.filter((product) => {
    const matchesSearch = !searchQuery ||
      product.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !selectedCategory || product.category?.id === Number(selectedCategory);
    return matchesSearch && matchesCategory;
  });

  const displayedRecommendations = showRecommendations && recommendations.length > 0
    ? recommendations.slice(0, sectionLimits.main)
    : [];
  const crossSellProducts = showRecommendations && recommendations.length > sectionLimits.main
    ? recommendations.slice(sectionLimits.main, sectionLimits.main + sectionLimits.crossSell)
    : [];
  const frequentlyBoughtTogether = showRecommendations && recommendations.length > sectionLimits.main + sectionLimits.crossSell
    ? recommendations.slice(sectionLimits.main + sectionLimits.crossSell, sectionLimits.main + sectionLimits.crossSell + sectionLimits.frequentlyBought)
    : [];

  const selectedUser = users.find(u => String(u.id) === String(selectedUserId));

  if (isInitializing) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '100vh',
        fontSize: '18px',
        color: '#666',
        background: '#eaeded',
        flexDirection: 'column'
      }}>
        <div style={{
          width: '50px',
          height: '50px',
          border: '4px solid #f3f3f3',
          borderTop: '4px solid #6366f1',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          marginBottom: '20px'
        }}></div>
        <p>Loading application...</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="main-header">
        <div className="header-top">
          <div className="header-left">
            <div className="logo">ShopHub</div>
            <div className="delivery-location">
              <span className="delivery-text">Hello,</span>
              <span className="location-text">{selectedUser?.name || "Guest"}</span>
            </div>
          </div>
          
          <div className="search-container">
            <select className="search-category">
              <option>All</option>
              {categories.map((cat) => (
                <option key={cat.id} value={cat.id}>{cat.name}</option>
              ))}
            </select>
            <input
              type="text"
              className="search-input"
              placeholder="Search products..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
            />
            <button className="search-button">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </button>
          </div>

          <div className="header-right">
            <div className="user-selector-header">
              <select
                value={selectedUserId}
                onChange={(e) => setSelectedUserId(e.target.value)}
                className="user-select"
              >
                {users.length === 0 && <option>No users</option>}
                {users.map((u) => (
                  <option key={u.id} value={String(u.id)}>
                    {u.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="cart-icon">
              <span className="cart-count">{cart.length}</span>
              <span>Cart</span>
            </div>
          </div>
        </div>

        <nav className="header-nav">
          <div className="nav-left">
            <button 
              className={`nav-button ${!selectedCategory ? "active" : ""}`}
              onClick={() => setSelectedCategory("")}
            >
              All
            </button>
            {categories.map((cat) => (
              <button
                key={cat.id}
                className={`nav-button ${selectedCategory === String(cat.id) ? "active" : ""}`}
                onClick={() => setSelectedCategory(selectedCategory === String(cat.id) ? "" : String(cat.id))}
              >
                {cat.name}
              </button>
            ))}
          </div>
          <button 
            className="get-recommendations-btn" 
            onClick={handleGetRecommendations} 
            disabled={loadingRec || !selectedUserId}
          >
            {loadingRec ? (
              <>
                <span className="loading-spinner"></span>
                <span style={{ marginLeft: "8px" }}>Loading...</span>
              </>
            ) : (
              "Get Recommendations"
            )}
          </button>
        </nav>
      </header>

      {error && (
        <div className="error-banner">
          {error}
          <button 
            onClick={() => setError("")} 
            style={{ 
              marginLeft: '10px', 
              background: 'transparent', 
              border: 'none', 
              color: 'white', 
              cursor: 'pointer',
              fontSize: '18px',
              padding: '0 5px'
            }}
            title="Dismiss"
          >
            ×
          </button>
        </div>
      )}

      <main className="main-content">
        {showRecommendations && displayedRecommendations.length > 0 && (
          <>
            <section className="recommendation-section">
              <h2 className="section-title">⭐ Recommended for You</h2>
              {loadingRec ? (
                <div className="loading-container">
                  <div className="loading-spinner large"></div>
                  <p className="loading-text">Finding the perfect products for you...</p>
                </div>
              ) : (
                <div className="product-row">
                  {displayedRecommendations.map((rec) => (
                    <ProductCard
                      key={rec.product_id}
                      product={rec}
                      onView={() => handleProductClick(rec.product_id)}
                      onAddToCart={() => handleAddToCart(rec.product_id)}
                      onPurchase={() => handlePurchase(rec.product_id)}
                      onWishlistToggle={() => handleWishlistToggle(rec.product_id)}
                      isInWishlist={wishlist.some((item) => item.product_id === rec.product_id)}
                      explanation={rec.explanation}
                      stockThreshold={stockThreshold}
                      descriptionMaxLength={descriptionMaxLength}
                    />
                  ))}
                </div>
              )}
            </section>

            {crossSellProducts.length > 0 && (
              <section className="recommendation-section">
                <h2 className="section-title">👥 Customers who viewed this item also viewed</h2>
                <div className="product-row">
                  {crossSellProducts.map((rec) => (
                    <ProductCard
                      key={rec.product_id}
                      product={rec}
                      onView={() => handleProductClick(rec.product_id)}
                      onAddToCart={() => handleAddToCart(rec.product_id)}
                      onPurchase={() => handlePurchase(rec.product_id)}
                      onWishlistToggle={() => handleWishlistToggle(rec.product_id)}
                      isInWishlist={wishlist.some((item) => item.product_id === rec.product_id)}
                      stockThreshold={stockThreshold}
                      descriptionMaxLength={descriptionMaxLength}
                    />
                  ))}
                </div>
              </section>
            )}

            {frequentlyBoughtTogether.length > 0 && (
              <section className="recommendation-section">
                <h2 className="section-title">🛒 Frequently bought together</h2>
                <div className="product-row">
                  {frequentlyBoughtTogether.map((rec) => (
                    <ProductCard
                      key={rec.product_id}
                      product={rec}
                      onView={() => handleProductClick(rec.product_id)}
                      onAddToCart={() => handleAddToCart(rec.product_id)}
                      onPurchase={() => handlePurchase(rec.product_id)}
                      onWishlistToggle={() => handleWishlistToggle(rec.product_id)}
                      isInWishlist={wishlist.some((item) => item.product_id === rec.product_id)}
                      stockThreshold={stockThreshold}
                      descriptionMaxLength={descriptionMaxLength}
                    />
                  ))}
                </div>
              </section>
            )}
          </>
        )}

        <section className="product-section">
          <h2 className="section-title">
            {searchQuery 
              ? `🔍 Search results for "${searchQuery}"` 
              : "🛍️ Shop by Category"}
            {filteredProducts.length > 0 && (
              <span className="product-count">
                ({filteredProducts.length} {filteredProducts.length === 1 ? 'product' : 'products'})
              </span>
            )}
          </h2>
          {filteredProducts.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🔍</div>
              <div className="empty-state-title">No products found</div>
              <div className="empty-state-subtitle">
                {searchQuery ? "Try a different search term" : products.length === 0 ? "No products in database. Please seed data." : "Select a category to browse products"}
              </div>
            </div>
          ) : (
            <div className="product-grid">
              {filteredProducts.map((product) => (
                <ProductCard
                  key={product.id}
                  product={{
                    product_id: product.id,
                    product_name: product.name,
                    category: product.category?.name || 'Uncategorized',
                    price: product.price,
                    description: product.description,
                    image_url: product.image_url,
                    average_rating: product.average_rating || 0,
                    rating_count: product.rating_count || 0,
                    stock: product.stock || 0,
                  }}
                  onView={() => handleProductClick(product.id)}
                  onAddToCart={() => handleAddToCart(product.id)}
                  onPurchase={() => handlePurchase(product.id)}
                  onWishlistToggle={() => handleWishlistToggle(product.id)}
                  isInWishlist={wishlist.some((item) => item.product_id === product.id)}
                  stockThreshold={stockThreshold}
                  descriptionMaxLength={descriptionMaxLength}
                />
              ))}
            </div>
          )}
        </section>
      </main>

      <footer className="main-footer">
        <div className="footer-content">
          <p>&copy; {new Date().getFullYear()} ShopHub. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

function ProductCard({
  product,
  onView,
  onAddToCart,
  onPurchase,
  onWishlistToggle,
  isInWishlist,
  explanation,
  stockThreshold = 10,
  descriptionMaxLength = 120,
}) {
  const averageRating = product.average_rating || 0;
  const ratingCount = product.rating_count || 0;
  const maxStars = 5;
  const fullStars = Math.floor(averageRating);
  const hasHalfStar = averageRating % 1 >= 0.5;
  const emptyStars = maxStars - fullStars - (hasHalfStar ? 1 : 0);

  const stockStatus = product.stock > 0 
    ? (product.stock > stockThreshold 
        ? "In Stock" 
        : `Only ${product.stock} left`)
    : "Out of Stock";

  const truncatedDescription = product.description && product.description.length > descriptionMaxLength
    ? product.description.substring(0, descriptionMaxLength) + "..."
    : product.description;

  return (
    <div className="product-card" onClick={onView}>
      <div className="product-image-wrapper">
        {product.image_url ? (
          <>
            <img
              src={product.image_url}
              alt={product.product_name || 'Product'}
              className="product-image"
              onError={(e) => {
                e.target.style.display = 'none';
                const placeholder = e.target.nextElementSibling;
                if (placeholder) {
                  placeholder.style.display = 'flex';
                }
              }}
              loading="lazy"
            />
            <div className="product-image-placeholder" style={{ display: 'none' }}>
              <span>No Image Available</span>
            </div>
          </>
        ) : (
          <div className="product-image-placeholder">
            <span>No Image Available</span>
          </div>
        )}
        {explanation && (
          <div className="recommendation-badge">Recommended</div>
        )}
      </div>
      <div className="product-info">
        <p className="product-title">{product.product_name || 'Untitled Product'}</p>
        <div className="product-rating">
          <span className="stars">
            {'★'.repeat(fullStars)}
            {hasHalfStar && '½'}
            {'☆'.repeat(emptyStars)}
          </span>
          <span className="rating-count">
            ({ratingCount > 0 ? ratingCount : 'No ratings'})
          </span>
        </div>
        <div className="product-price">
          <span className="price-symbol">₹</span>
          <span className="price-amount">
            {(product.price || 0).toLocaleString('en-IN', { 
              minimumFractionDigits: 0, 
              maximumFractionDigits: 0 
            })}
          </span>
        </div>
        {product.stock !== undefined && (
          <div className={`stock-status ${product.stock > 0 ? 'in-stock' : 'out-of-stock'}`}>
            {stockStatus}
          </div>
        )}
        {truncatedDescription && (
          <p className="product-description">
            {truncatedDescription}
          </p>
        )}
        <div className="product-actions">
          <button
            className="add-to-cart-btn"
            onClick={(e) => {
              e.stopPropagation();
              onAddToCart();
            }}
            disabled={product.stock === 0}
          >
            Add to Cart
          </button>
          <button
            className="buy-now-btn"
            onClick={(e) => {
              e.stopPropagation();
              onPurchase();
            }}
            disabled={product.stock === 0}
          >
            Buy Now
          </button>
        </div>
        <button
          className={`wishlist-btn ${isInWishlist ? "active" : ""}`}
          onClick={(e) => {
            e.stopPropagation();
            onWishlistToggle();
          }}
          title={isInWishlist ? "Remove from wishlist" : "Add to wishlist"}
        >
          {isInWishlist ? "❤️" : "🤍"}
        </button>
        {explanation && (
          <div className="recommendation-explanation">
            <strong>Why we recommend this:</strong> {explanation}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

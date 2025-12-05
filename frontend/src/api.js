import axios from "axios";

// Dynamic baseURL from environment variable or default (Vite uses import.meta.env)
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
});

// GET /api/users/
export async function fetchUsers() {
  const res = await api.get("/users/");
  return res.data.users || [];
}

// GET /api/products/
export async function fetchProducts() {
  const res = await api.get("/products/");
  return res.data.products || [];
}

// GET /api/categories/
export async function fetchCategories() {
  const res = await api.get("/categories/");
  return res.data.categories || [];
}

// GET /api/recommendations/user/{user_id}/
export async function fetchRecommendations(userId, limit = null) {
  const params = {};
  if (limit !== null) {
    params.limit = limit;
  }
  const res = await api.get(`/recommendations/user/${userId}/`, { params });
  return res.data;
}

// POST /api/interactions/
export async function createInteraction(payload) {
  const res = await api.post("/interactions/", payload);
  return res.data;
}

// POST /api/browsing/
export async function trackBrowsing(userId, productId) {
  const res = await api.post("/browsing/", {
    user_id: userId,
    product_id: productId,
  });
  return res.data;
}

// POST /api/search/
export async function trackSearch(userId, query) {
  const res = await api.post("/search/", {
    user_id: userId,
    query: query,
  });
  return res.data;
}

// POST /api/wishlist/
export async function addToWishlist(userId, productId) {
  const res = await api.post("/wishlist/", {
    user_id: userId,
    product_id: productId,
  });
  return res.data;
}

// DELETE /api/wishlist/{user_id}/{product_id}/
export async function removeFromWishlist(userId, productId) {
  const res = await api.delete(`/wishlist/${userId}/${productId}/`);
  return res.data;
}

// GET /api/wishlist/{user_id}/
export async function getWishlist(userId) {
  const res = await api.get(`/wishlist/${userId}/`);
  return res.data.items || [];
}

export default api;

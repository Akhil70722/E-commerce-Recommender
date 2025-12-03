import axios from "axios";

// Adjust baseURL if your backend runs on a different host/port.
const api = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
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

// GET /api/recommendations/user/{user_id}/
export async function fetchRecommendations(userId) {
  const res = await api.get(`/recommendations/user/${userId}/`);
  return res.data;
}

// POST /api/interactions/
export async function createInteraction(payload) {
  const res = await api.post("/interactions/", payload);
  return res.data;
}

export default api;



import { useEffect, useState } from "react";
import {
  fetchUsers,
  fetchProducts,
  fetchRecommendations,
  createInteraction,
} from "./api";
import "./App.css";

function App() {
  const [users, setUsers] = useState([]);
  const [products, setProducts] = useState([]);
  const [selectedUserId, setSelectedUserId] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loadingRec, setLoadingRec] = useState(false);
  const [error, setError] = useState("");

  const [interactionProductId, setInteractionProductId] = useState("");
  const [interactionType, setInteractionType] = useState("view");
  const [interactionRating, setInteractionRating] = useState("");

  useEffect(() => {
    async function loadInitialData() {
      try {
        setError("");
        const [u, p] = await Promise.all([fetchUsers(), fetchProducts()]);
        setUsers(u);
        setProducts(p);
        if (u.length > 0) setSelectedUserId(u[0].id.toString());
      } catch (err) {
        console.error(err);
        setError("Failed to load users/products. Is the backend running?");
      }
    }
    loadInitialData();
  }, []);

  async function handleGetRecommendations() {
    if (!selectedUserId) {
      setError("Please select a user first.");
      return;
    }
    try {
      setLoadingRec(true);
      setError("");
      const data = await fetchRecommendations(selectedUserId);
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.error(err);
      setError(
        "Failed to fetch recommendations. Check backend logs and GROQ_API_KEY."
      );
    } finally {
      setLoadingRec(false);
    }
  }

  async function handleCreateInteraction(e) {
    e.preventDefault();
    if (!selectedUserId || !interactionProductId) {
      setError("Select user and product for interaction.");
      return;
    }
    try {
      setError("");
      const payload = {
        user_id: Number(selectedUserId),
        product_id: Number(interactionProductId),
        interaction_type: interactionType,
        rating:
          interactionType === "rating" && interactionRating
            ? Number(interactionRating)
            : null,
      };
      await createInteraction(payload);
      // Simple confirmation; in a real app you'd refresh interactions.
      alert("Interaction recorded. Now click 'Fetch Recommendations'.");
    } catch (err) {
      console.error(err);
      setError("Failed to create interaction.");
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-top">
          <div className="brand">
            <h1 className="app-title">E-commerce Product Recommender</h1>
            <p className="app-subtitle">
              Personalized product suggestions with transparent LLM
              explanations.
            </p>
            <div className="tag-row">
              <span className="badge-pill">Live Demo</span>
              <span className="tag">Django · PostgreSQL</span>
              <span className="tag">Groq LLM Explanations</span>
              <span className="tag">React Frontend</span>
            </div>
          </div>
        </div>
      </header>

      {error && <div className="error">{error}</div>}

      <div className="layout">
        <section className="panel">
          <h2>1. Select User</h2>
          <p className="panel-subtitle">
            Choose a shopper profile to see how recommendations adapt.
          </p>
          <select
            value={selectedUserId}
            onChange={(e) => setSelectedUserId(e.target.value)}
          >
            {users.map((u) => (
              <option key={u.id} value={u.id}>
                {u.name} ({u.email})
              </option>
            ))}
          </select>

          <h2 style={{ marginTop: "1.5rem" }}>2. Record Interaction</h2>
          <p className="panel-subtitle">
            Simulate browsing behaviour to influence future recommendations.
          </p>
          <form className="interaction-form" onSubmit={handleCreateInteraction}>
            <label>
              Product:
              <select
                value={interactionProductId}
                onChange={(e) => setInteractionProductId(e.target.value)}
              >
                <option value="">-- select product --</option>
                {products.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.category.name})
                  </option>
                ))}
              </select>
            </label>

            <label>
              Type:
              <select
                value={interactionType}
                onChange={(e) => setInteractionType(e.target.value)}
              >
                <option value="view">View</option>
                <option value="cart">Add to Cart</option>
                <option value="purchase">Purchase</option>
                <option value="rating">Rating</option>
              </select>
            </label>

            {interactionType === "rating" && (
              <label>
                Rating (1–5):
                <input
                  type="number"
                  min="1"
                  max="5"
                  value={interactionRating}
                  onChange={(e) => setInteractionRating(e.target.value)}
                />
              </label>
            )}

            <button type="submit">Save Interaction</button>
          </form>
          <p className="hint">
            More interactions → better recommendations tailored to this user.
          </p>
        </section>

        <section className="panel">
          <h2>3. Recommendations</h2>
          <p className="panel-subtitle">
            Generated from user behaviour, product content, and Groq LLM
            explanations.
          </p>

          <button onClick={handleGetRecommendations} disabled={loadingRec}>
            {loadingRec ? "Loading..." : "Fetch Recommendations"}
          </button>

          <div className="recommendations">
            {recommendations.length === 0 && !loadingRec && (
              <p className="empty-state">
                No recommendations yet. Record a few interactions above and
                refresh.
              </p>
            )}

            {recommendations.map((rec) => (
              <div key={rec.product_id} className="card">
                <div className="card-header">
                  <h3>{rec.product_name}</h3>
                  <span className="category">{rec.category}</span>
                </div>
                <p className="price">${rec.price.toFixed(2)}</p>
                <p className="description">{rec.description}</p>
                <p className="score">Score: {rec.score}</p>
                <p className="explanation">
                  <strong>Why this product:</strong> {rec.explanation}
                </p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;

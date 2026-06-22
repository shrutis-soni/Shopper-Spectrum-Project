import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# ── Page config 
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models 
@st.cache_resource
def load_models():
    kmeans     = joblib.load("kmeans_model.pkl")
    scaler     = joblib.load("rfm_scaler.pkl")
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    item_sim_df = pd.read_pickle("item_similarity.pkl")
    with open("product_list.pkl", "rb") as f:
        products = pickle.load(f)
    return kmeans, scaler, label_map, item_sim_df, products

kmeans, scaler, label_map, item_sim_df, products = load_models()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    page = st.session_state.get("page", "Home")

    def nav_button(label, icon, key):
        active_class = "active" if st.session_state.get("page") == key else ""
        if st.button(f"{icon}  {label}", key=f"nav_{key}",
                     use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    nav_button("Home",           "🖥",  "Home")
    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    nav_button("Clustering",     "📅",  "Clustering")
    nav_button("Recommendation", "📊",  "Recommendation")

page = st.session_state.get("page", "Home")

# ── Helper functions ───────────────────────────────────────────────────────────
def predict_segment(recency, frequency, monetary):
    rfm_arr = np.array([[recency, frequency, monetary]], dtype=float)
    rfm_log = np.log1p(rfm_arr)
    rfm_scaled = scaler.transform(rfm_log)
    cluster = kmeans.predict(rfm_scaled)[0]
    segment = label_map.get(int(cluster), "Unknown")
    return int(cluster), segment

def get_recommendations(product_name: str, top_n: int = 5):
    product_upper = product_name.strip().upper()
    matches = [p for p in item_sim_df.index if product_upper in p.upper()]
    if not matches:
        return None, []
    best_match = matches[0]
    scores = item_sim_df[best_match].drop(index=best_match).sort_values(ascending=False)
    return best_match, scores.head(top_n).index.tolist()

# ── Pages ──────────────────────────────────────────────────────────────────────

# HOME
if page == "Home":
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("### Customer Segmentation & Product Recommendation System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📊 About This App")
        st.write(
            "Shopper Spectrum analyzes e-commerce transaction data to uncover "
            "customer behavior patterns and deliver personalized experiences."
        )
    with col2:
        st.markdown("#### 📅 Clustering")
        st.write(
            "Enter a customer's Recency, Frequency, and Monetary values to "
            "predict their segment: **High-Value**, **Regular**, **Occasional**, or **At-Risk**."
        )
    with col3:
        st.markdown("#### 📋 Recommendation")
        st.write(
            "Type a product name to get **5 similar product recommendations** "
            "using item-based collaborative filtering and cosine similarity."
        )

    st.markdown("---")
    st.markdown("**Dataset**: UCI Online Retail (2022–2023) · **Algorithm**: KMeans + Cosine Similarity")

# CLUSTERING
elif page == "Clustering":
    st.markdown("## Customer Segmentation")
    st.markdown("")

    recency   = st.number_input("Recency (days since last purchase)",
                                 min_value=0, max_value=1000, value=30, step=1)
    frequency = st.number_input("Frequency (number of purchases)",
                                 min_value=1, max_value=500, value=5, step=1)
    monetary  = st.number_input("Monetary (total spend)",
                                 min_value=0.0, max_value=1_000_000.0,
                                 value=500.0, step=1.0, format="%.2f")

    st.markdown("")
    if st.button("Predict Segment"):
        cluster, segment = predict_segment(recency, frequency, monetary)
        st.markdown(f"<p class='result-cluster'>{cluster}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p class='result-segment'>This customer belongs to: <b>{segment}</b></p>",
            unsafe_allow_html=True
        )

        # Segment description
        descriptions = {
            "High-Value":  "🏆 Recent, frequent, and high-spending customer. Focus on loyalty rewards.",
            "Regular":     "🔁 Steady buyer with moderate spend. Good upsell candidate.",
            "Occasional":  "🕐 Infrequent purchaser. Target with re-engagement campaigns.",
            "At-Risk":     "⚠️ Has not purchased recently. Consider win-back offers.",
        }
        st.info(descriptions.get(segment, ""))

# RECOMMENDATION
elif page == "Recommendation":
    st.markdown("## Product Recommender")
    st.markdown("")

    st.markdown("Enter Product Name")
    product_input = st.text_input("", placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
                                   label_visibility="collapsed")

    st.markdown("")
    if st.button("Recommend"):
        if not product_input.strip():
            st.warning("Please enter a product name.")
        else:
            matched, recs = get_recommendations(product_input)
            if not recs:
                st.error(f"No product matching '{product_input}' found in the catalog.")
            else:
                st.markdown(f"**Showing recommendations for:** `{matched}`")
                st.markdown("**Recommended Products:**")
                st.markdown("")
                for rec in recs:
                    st.markdown(
                        f"<p class='rec-item'>{rec}</p>",
                        unsafe_allow_html=True
                    )

    # Show sample products hint
    with st.expander("💡 Sample product names you can try"):
        samples = [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "GREEN VINTAGE SPOT BEAKER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
            "PARTY BUNTING",
            "PACK OF 72 RETROSPOT CAKE CASES",
        ]
        for s in samples:
            st.code(s, language=None)
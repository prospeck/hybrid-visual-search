from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.getenv("VISUAL_SEARCH_API_URL", "http://localhost:8020").rstrip("/")

st.set_page_config(page_title="Hybrid Visual Search", page_icon="VIS", layout="wide")
st.title("Hybrid Visual Search")
st.caption("CLIP semantic retrieval fused with YOLO object constraints")

try:
    health = httpx.get(f"{API_URL}/health", timeout=10).json()
except Exception:
    st.error(f"API unavailable at {API_URL}")
    st.stop()

with st.sidebar:
    st.header("Index")
    st.metric("Indexed images", health["indexed_images"])
    st.caption(
        f"{health['detector_backend']} + {health['encoder_backend']} | "
        f"{health['device']} | {health['vector_backend']}"
    )
    upload = st.file_uploader("Upload and index an image", type=["jpg", "jpeg", "png", "webp"])
    if upload and st.button("Index upload", type="primary", width="stretch"):
        with st.spinner("Detecting objects and generating CLIP embedding..."):
            response = httpx.post(
                f"{API_URL}/v1/index/upload",
                files={"file": (upload.name, upload.getvalue(), upload.type)},
                timeout=180,
            )
        if response.is_success:
            st.success("Image indexed")
            st.rerun()
        else:
            st.error(response.text)

query = st.text_input(
    "Semantic query",
    placeholder="people cycling in a city, a dog on the beach, food on a table...",
)

try:
    available_classes = httpx.get(f"{API_URL}/v1/classes", timeout=10).json()
except Exception:
    available_classes = []

with st.expander("Object constraints", expanded=True):
    selected_classes = st.multiselect("YOLO classes", available_classes)
    object_mode = st.radio("Combine filters", ["all", "any"], horizontal=True)
    constraints = []
    if selected_classes:
        columns = st.columns(len(selected_classes))
        for column, label in zip(columns, selected_classes, strict=True):
            with column:
                minimum = st.number_input(f"Minimum {label}", 0, 100, 1, key=f"min-{label}")
                maximum_enabled = st.checkbox(f"Cap {label}", key=f"cap-{label}")
                maximum = (
                    st.number_input(
                        f"Maximum {label}", int(minimum), 100, int(minimum), key=f"max-{label}"
                    )
                    if maximum_enabled
                    else None
                )
                constraints.append({"label": label, "min_count": minimum, "max_count": maximum})

left, middle, right = st.columns(3)
top_k = left.slider("Results", 1, 50, 12)
semantic_weight = middle.slider("Semantic weight", 0.0, 1.0, 0.75, 0.05)
annotated = right.toggle("Draw YOLO boxes", value=True)

if st.button("Search", type="primary", width="stretch"):
    if not query and not constraints:
        st.warning("Enter a semantic query, add an object constraint, or use both.")
    else:
        with st.spinner("Searching FAISS and applying object filters..."):
            response = httpx.post(
                f"{API_URL}/v1/search",
                json={
                    "query": query,
                    "objects": constraints,
                    "object_mode": object_mode,
                    "top_k": top_k,
                    "semantic_weight": semantic_weight,
                },
                timeout=120,
            )
        if not response.is_success:
            st.error(response.text)
        else:
            payload = response.json()
            st.caption(
                f"{len(payload['hits'])} results from {payload['total_candidates']} candidates | "
                f"{payload['latency_ms']:.1f} ms | {payload['vector_backend']}"
            )
            grid = st.columns(3)
            for index, hit in enumerate(payload["hits"]):
                with grid[index % 3]:
                    image_url = (
                        f"{API_URL}{hit['image_url']}?annotated={'true' if annotated else 'false'}"
                    )
                    st.image(image_url, width="stretch")
                    st.markdown(f"**{hit['filename']}**")
                    st.caption(
                        f"hybrid {hit['score']:.3f} | semantic {hit['semantic_score']:.3f} | "
                        f"objects {hit['object_score']:.3f}"
                    )
                    st.json(hit["object_counts"], expanded=False)

with st.expander("System design notes"):
    st.markdown(
        "The model never searches filenames. CLIP maps text and pixels into a shared vector "
        "space; FAISS retrieves nearest images; YOLO counts enforce exact object constraints. "
        "The final score is a configurable fusion of both signals."
    )

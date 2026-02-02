import streamlit as st
import json
import os
import random
from datetime import datetime
from PIL import Image

DATA_PATH = "data/samples.json"
ANNOTATIONS_DIR = "annotations"

# ---------------------------
# Utilities
# ---------------------------

def load_samples():
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def get_existing_annotators():
    if not os.path.exists(ANNOTATIONS_DIR):
        return []
    return [
        f.replace(".json", "")
        for f in os.listdir(ANNOTATIONS_DIR)
        if f.endswith(".json")
    ]

def get_annotation_path(annotator_id):
    return os.path.join(ANNOTATIONS_DIR, f"{annotator_id}.json")

def load_annotations(annotator_id):
    path = get_annotation_path(annotator_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_all_annotations(annotator_id, annotations):
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    path = get_annotation_path(annotator_id)
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)

def load_text(path):
    with open(path, "r") as f:
        return f.read()

# ---------------------------
# Page config
# ---------------------------

st.set_page_config(layout="wide")
st.title("Anomaly Annotation Tool")

# ---------------------------
# Annotator selection gate
# ---------------------------

if "annotator_id" not in st.session_state:
    existing = get_existing_annotators()

    st.subheader("Select annotator")

    if "annotator_mode" not in st.session_state:
        st.session_state.annotator_mode = "existing"

    st.session_state.annotator_mode = st.radio(
        "Annotator type",
        ["existing", "new"],
        format_func=lambda x: "Existing annotator" if x == "existing" else "Create new annotator",
    )

    annotator_id = None

    if st.session_state.annotator_mode == "existing":
        if existing:
            annotator_id = st.selectbox("Choose annotator", existing)
        else:
            st.info("No existing annotators found. Please create a new one.")
    else:
        annotator_id = st.text_input("New annotator ID", placeholder="e.g. A3")

    if st.button("Continue"):
        if not annotator_id:
            st.warning("Please provide an annotator ID")
            st.stop()

        st.session_state.annotator_id = annotator_id
        st.session_state.pop("annotator_mode", None)
        st.rerun()

    st.stop()

annotator_id = st.session_state.annotator_id
st.caption(f"Annotator: **{annotator_id}**")

# ---------------------------
# Session initialization
# ---------------------------

if "initialized" not in st.session_state:
    samples = load_samples()
    existing_annotations = load_annotations(annotator_id)
    annotated_ids = {a["sample_id"] for a in existing_annotations}

    remaining = [s for s in samples if s["id"] not in annotated_ids]
    random.shuffle(remaining)

    st.session_state.sample_order = remaining
    st.session_state.total_samples = len(samples)          # <-- FIXED
    st.session_state.current_idx = 0
    st.session_state.annotations_buffer = {
        a["sample_id"]: a for a in existing_annotations
    }
    st.session_state.review_mode = False
    st.session_state.initialized = True

# ---------------------------
# Progress bar
# ---------------------------

done = len(st.session_state.annotations_buffer)
total = st.session_state.total_samples

st.markdown(f"**Progress:** {done} / {total}")
st.progress(done / total if total > 0 else 0.0)
# <-- ADD THIS BLOCK BACK (TOP OF PAGE) -->
if len(st.session_state.annotations_buffer) == len(st.session_state.sample_order) and not st.session_state.review_mode:
    if st.button("📋 Review all annotations"):
        st.session_state.review_mode = True
        st.rerun()
# ---------------------------
# Review mode
# ---------------------------

if st.session_state.review_mode:
    st.subheader("Final Review")
    st.markdown("Click a sample to edit it:")

    for i, sample in enumerate(st.session_state.sample_order):
        ann = st.session_state.annotations_buffer.get(sample["id"])
        label = f"📝 Sample {i+1}: {sample['id']}"
        if ann is None:
            label = "⚠️ " + label

        if st.button(label, key=f"review_{i}"):
            st.session_state.current_idx = i
            st.session_state.review_mode = False
            st.rerun()

    st.divider()

    if st.button("✅ Final submit"):
        save_all_annotations(
            annotator_id,
            list(st.session_state.annotations_buffer.values())
        )
        st.success("All annotations submitted 🎉")
        st.session_state.clear()
        st.stop()

    st.stop()

# ---------------------------
# Annotation mode
# ---------------------------

samples = st.session_state.sample_order

if not samples:
    st.success("All samples annotated 🎉")
    st.stop()

idx = min(st.session_state.current_idx, len(samples) - 1)
st.session_state.current_idx = idx

sample = samples[idx]
existing = st.session_state.annotations_buffer.get(sample["id"], {})

anomaly = sample.get("anomaly_label", 0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(Image.open(sample["image"]), use_container_width=True)
    st.text("There is an anomaly" if anomaly else "No anomaly")

with col2:
    st.subheader("VLM Output")
    st.text(load_text(sample["vlm_output"]))

st.divider()
st.subheader("Annotation")

anomaly_presence = st.radio(
    "Does the model correctly identify whether there is an anomaly?",
    ["yes", "no", "unsure"],
    index=["yes", "no", "unsure"].index(
        existing.get("anomaly_presence", "yes")
    ),
)

type_correctness = (
    st.radio(
        "Is the anomaly type correctly identified?",
        ["correct", "partial", "incorrect"],
        index=["correct", "partial", "incorrect"].index(
            existing.get("type_correctness", "correct")
        ),
    )
    if anomaly else "N/A"
)

localization_score = (
    st.slider(
        "How well does the model identify the location of the anomaly?",
        1, 5,
        value=existing.get("localization_score", 3),
    )
    if anomaly else "N/A"
)

grounded_reasoning = st.slider(
    "Is the explanation grounded in visual evidence?",
    1, 5,
    value=existing.get("grounded_reasoning", 3),
)

col_prev, _, col_next = st.columns([1, 12, 3])

with col_prev:
    if st.button("⬅ Back") and idx > 0:
        st.session_state.current_idx -= 1
        st.rerun()

with col_next:
    if st.button("💾 Save and continue"):
        st.session_state.annotations_buffer[sample["id"]] = {
            "sample_id": sample["id"],
            "annotator_id": annotator_id,
            "anomaly_presence": anomaly_presence,
            "type_correctness": type_correctness,
            "localization_score": localization_score,
            "grounded_reasoning": grounded_reasoning,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if idx < len(samples) - 1:
            st.session_state.current_idx += 1
        else:
            st.session_state.review_mode = True

        st.rerun()

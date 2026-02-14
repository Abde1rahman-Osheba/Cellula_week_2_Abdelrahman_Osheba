import streamlit as st
from PIL import Image

from imagecaption import generate_caption
from text_classifier import TextClassifier
from database import save_record, load_records

st.set_page_config(
    page_title="Text and Image Classifier",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache the classifier so it loads only once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification models. This may take a few minutes on first run.")
def get_classifier():
    return TextClassifier()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Classify Text", "Classify Image", "View Database"],
)


# ===========================================================================
# PAGE 1 — Classify Text
# ===========================================================================
if page == "Classify Text":
    st.title("Text Classification")
    st.write("Enter text below to classify using the hybrid ensemble model.")

    user_text = st.text_area(
        "Input Text",
        height=150,
        placeholder="Type or paste text here...",
    )

    if st.button("Classify"):
        if not user_text.strip():
            st.warning("Please enter text before submitting.")
        else:
            classifier = get_classifier()
            with st.spinner("Processing..."):
                result = classifier.classify(query=user_text)

            st.success(f"Classification: {result['label']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

            st.subheader("Class Probabilities")
            st.bar_chart(result["probs"])

            save_record(
                input_type="text",
                input_text=user_text.strip(),
                classification_label=result["label"],
                confidence=result["confidence"],
            )

            st.info("Result saved successfully.")


# ===========================================================================
# PAGE 2 — Classify Image
# ===========================================================================
elif page == "Classify Image":
    st.title("Image Classification")
    st.write(
        "Upload an image."
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Generate Caption and Classify"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image)

            st.subheader("Generated Caption")
            st.write(caption)

            classifier = get_classifier()
            with st.spinner("Classifying caption..."):
                result = classifier.classify(query="", image_caption=caption)

            st.success(f"Classification: {result['label']}")
            st.metric("Confidence", f"{result['confidence']:.2%}")

            st.subheader("Class Probabilities")
            st.bar_chart(result["probs"])

            save_record(
                input_type="image",
                input_text=caption,
                classification_label=result["label"],
                confidence=result["confidence"],
            )

            st.info("Result saved successfully.")


# ===========================================================================
# PAGE 3 — View Database
# ===========================================================================
elif page == "View Database":
    st.title("Classification Records")
    st.write("All previous classification inputs and results are stored below.")

    records = load_records()

    if records.empty:
        st.info("No records available.")
    else:
        st.metric("Total Records", len(records))
        st.dataframe(records, use_container_width=True)

        csv_data = records.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="classification_records.csv",
            mime="text/csv",
        )

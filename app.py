
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Breast Cancer Classification")
st.write(
    "An interactive application to compare five "
    "machine learning classification models."
)

st.info(
    "This application is for educational purposes only "
    "and is not intended for medical diagnosis."
)

# --------------------------------------------------
# Load Trained Models
# --------------------------------------------------

MODEL_DIR = "model"

@st.cache_resource
def load_models():

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "SVM": "svm.joblib"
    }

    loaded_models = {}

    for name, filename in model_files.items():

        path = os.path.join(MODEL_DIR, filename)

        loaded_models[name] = joblib.load(path)

    return loaded_models


try:
    models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --------------------------------------------------
# Dataset Upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    # --------------------------------------------------
    # Validate Dataset
    # --------------------------------------------------

    if "target" not in df.columns:

        st.error(
            "The uploaded CSV must contain a target column."
        )

        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    # Load feature names from the training data
    expected_features = list(
        joblib.load(
            os.path.join(
                MODEL_DIR,
                "logistic_regression.joblib"
            )
        ).feature_names_in_
    )

    if set(X.columns) != set(expected_features):

        st.error(
            "Uploaded CSV does not contain the required "
            "30 input features."
        )

        st.write("Expected Features:", expected_features)

        st.stop()

    # Arrange features in the original training order
    X = X[expected_features]

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------

    st.header("2. Select Machine Learning Model")

    selected_model_name = st.selectbox(
        "Choose a model",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    # --------------------------------------------------
    # Make Predictions
    # --------------------------------------------------

    if st.button("Evaluate Model", type="primary"):

        y_pred = selected_model.predict(X)

        y_prob = selected_model.predict_proba(X)[:, 1]

        # --------------------------------------------------
        # Calculate Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y, y_pred)

        auc = roc_auc_score(y, y_prob)

        precision = precision_score(
            y, y_pred, zero_division=0
        )

        recall = recall_score(
            y, y_pred, zero_division=0
        )

        f1 = f1_score(
            y, y_pred, zero_division=0
        )

        mcc = matthews_corrcoef(y, y_pred)

        # --------------------------------------------------
        # Display Metrics
        # --------------------------------------------------

        st.header("3. Evaluation Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")

        col2.metric("AUC", f"{auc:.4f}")

        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")

        col5.metric("F1 Score", f"{f1:.4f}")

        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots()

        ax.imshow(cm)

        ax.set_title("Confusion Matrix")

        ax.set_xlabel("Predicted Label")

        ax.set_ylabel("Actual Label")

        ax.set_xticks([0, 1])

        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Malignant", "Benign"])

        ax.set_yticklabels(["Malignant", "Benign"])

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y,
            y_pred,
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4))

        # --------------------------------------------------
        # Actual vs Predicted
        # --------------------------------------------------

        st.header("6. Actual vs Predicted")

        results = pd.DataFrame({
            "Actual": y.values,
            "Predicted": y_pred
        })

        results["Actual Label"] = results["Actual"].map({
            0: "Malignant",
            1: "Benign"
        })

        results["Predicted Label"] = results["Predicted"].map({
            0: "Malignant",
            1: "Benign"
        })

        st.dataframe(results)

        st.success("Model evaluation completed successfully!")

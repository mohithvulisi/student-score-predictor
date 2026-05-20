from pathlib import Path
import io
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Score Predictor")
st.caption("Train a machine learning model on a student dataset and predict a target score.")

DEFAULT_DATA_PATH = Path("data/star98demo.csv")


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return df


def build_model(model_name: str, numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    if model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=2
        )
    else:
        model = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


with st.sidebar:
    st.header("1. Dataset")
    uploaded_file = st.file_uploader("Upload student CSV", type=["csv"])

    st.markdown("---")
    st.header("2. Model Settings")
    model_name = st.selectbox("Choose model", ["Random Forest", "Linear Regression"])
    test_size = st.slider("Test data size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    data_source = "uploaded CSV"
elif DEFAULT_DATA_PATH.exists():
    df = load_csv(DEFAULT_DATA_PATH)
    data_source = "data/star98.csv"
else:
    st.info(
        "Upload a CSV from the sidebar, or place your dataset at `data/star98.csv`."
    )
    st.stop()

df = clean_columns(df)

st.success(f"Loaded {data_source}: {df.shape[0]} rows × {df.shape[1]} columns")

tab_data, tab_train, tab_predict = st.tabs(["Dataset", "Train Model", "Predict"])

with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(30), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    col3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    with st.expander("Column Summary"):
        summary = pd.DataFrame({
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "unique_values": df.nunique(dropna=True).values
        })
        st.dataframe(summary, use_container_width=True)

with tab_train:
    st.subheader("Train a Regression Model")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("This app needs at least one numeric target column for score prediction.")
        st.stop()

    target_col = st.selectbox(
        "Choose the score/target column to predict",
        numeric_cols,
        index=len(numeric_cols) - 1
    )

    possible_features = [col for col in df.columns if col != target_col]

    default_features = possible_features[: min(8, len(possible_features))]
    selected_features = st.multiselect(
        "Choose input features",
        possible_features,
        default=default_features
    )

    if not selected_features:
        st.warning("Select at least one feature.")
        st.stop()

    working_df = df[selected_features + [target_col]].copy()
    working_df = working_df.dropna(subset=[target_col])

    if working_df.shape[0] < 20:
        st.warning("You need at least 20 usable rows for a meaningful train/test split.")
        st.stop()

    X = working_df[selected_features]
    y = working_df[target_col]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state)
    )

    model = build_model(model_name, numeric_features, categorical_features)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    st.session_state["trained_model"] = model
    st.session_state["selected_features"] = selected_features
    st.session_state["target_col"] = target_col
    st.session_state["training_columns"] = X.columns.tolist()
    st.session_state["training_df"] = X

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("R² Score", f"{r2:.3f}")

    st.markdown("#### Actual vs Predicted")
    result_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    }).reset_index(drop=True)
    st.dataframe(result_df.head(50), use_container_width=True)

    st.line_chart(result_df.head(50))

    model_bytes = pickle.dumps({
        "model": model,
        "features": selected_features,
        "target": target_col
    })

    st.download_button(
        label="Download trained model",
        data=model_bytes,
        file_name="student_score_model.pkl",
        mime="application/octet-stream"
    )

with tab_predict:
    st.subheader("Predict a Student Score")

    if "trained_model" not in st.session_state:
        st.info("Train the model first in the Train Model tab.")
        st.stop()

    model = st.session_state["trained_model"]
    selected_features = st.session_state["selected_features"]
    target_col = st.session_state["target_col"]
    training_df = st.session_state["training_df"]

    user_input = {}

    with st.form("prediction_form"):
        for col in selected_features:
            series = training_df[col]

            if pd.api.types.is_numeric_dtype(series):
                min_val = float(series.min()) if not pd.isna(series.min()) else 0.0
                max_val = float(series.max()) if not pd.isna(series.max()) else 100.0
                mean_val = float(series.mean()) if not pd.isna(series.mean()) else min_val

                if min_val == max_val:
                    user_input[col] = st.number_input(col, value=mean_val)
                else:
                    user_input[col] = st.slider(
                        col,
                        min_value=min_val,
                        max_value=max_val,
                        value=min(max(mean_val, min_val), max_val)
                    )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    user_input[col] = st.text_input(col, value="")
                else:
                    user_input[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("Predict Score")

    if submitted:
        input_df = pd.DataFrame([user_input])
        predicted_score = model.predict(input_df)[0]

        st.success(f"Predicted `{target_col}`: **{predicted_score:.2f}**")

        st.caption(
            "Note: This prediction is only as good as the dataset, selected features, and model quality."
        )

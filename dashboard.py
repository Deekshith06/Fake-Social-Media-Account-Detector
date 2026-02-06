

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from statistics import median
from sklearn.compose import ColumnTransformer
from twitter_fetcher import fetch_twitter_profile, extract_features_from_profile as extract_twitter_features
from instagram_fetcher import fetch_instagram_profile, extract_features_from_profile as extract_instagram_features

st.set_page_config(page_title="Fake Account Detector", layout="wide")

PROJECT_ROOT = os.getcwd()
DATASET = os.path.join(PROJECT_ROOT, "fake_dataset.xlsx")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "best_model.joblib")

st.title("Fake Social Media Account Detector")
st.markdown("Detect fake accounts using AI - Enter a username or manually input features.")

def detect_target(df):
    candidates = [c for c in df.columns if c.lower() in ("label","is_fake","fake","target","isbot","bot","class")]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if df[c].dropna().nunique() == 2:
            return c
    return None

@st.cache_data
def load_dataset(path):
    if not os.path.exists(path):
        return None
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        return pd.read_csv(path)

@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# -------------------------
# Load data + model early
# -------------------------
df = load_dataset(DATASET)
if df is None:
    st.warning(f"Dataset not found at {DATASET}. Place fake_dataset.xlsx in project root.")
    st.stop()

model = load_model(MODEL_PATH)  # may be None if model not trained yet

st.success(f"Loaded dataset: {os.path.basename(DATASET)} ({df.shape[0]} rows, {df.shape[1]} cols)")
st.info("Model loaded: " + ("Yes" if model is not None else "No (run pipeline first)"))

# detect target and derive feature columns (initial)
target = detect_target(df)
if target is None:
    st.error("Could not auto-detect target column. Add a column named 'is_fake'/'label' or pass --target in pipeline.")
    st.stop()

st.write(f"Detected target column: **{target}**")

# Build initial features list: drop ID-like and high-card text
drop_id_like = [c for c in df.columns if any(x in c.lower() for x in ("id","uuid","user_id","account_id","handle"))]
X_df = df.drop(columns=[target] + drop_id_like, errors='ignore')

# remove extremely high-card text columns ( > 50 unique ) — we will add model-required high-card cols back if needed
high_card_text = [c for c in X_df.select_dtypes(include=['object']).columns if X_df[c].nunique() > 50]
if high_card_text:
    X_df = X_df.drop(columns=high_card_text)

# If model exists, introspect it for required input columns and ensure they appear in X_df
def introspect_required_cols_from_pipeline(pipeline, df_example):
    required = []
    try:
        if pipeline is None:
            return required
        pre = None
        if hasattr(pipeline, 'named_steps') and 'pre' in pipeline.named_steps:
            pre = pipeline.named_steps['pre']
        else:
            if hasattr(pipeline, 'steps'):
                for n, step in pipeline.steps:
                    if isinstance(step, ColumnTransformer):
                        pre = step
                        break
        if pre is None:
            return required

        for name, transformer, cols in pre.transformers_:
            if isinstance(cols, (list, tuple, np.ndarray)):
                for c in cols:
                    if isinstance(c, str):
                        required.append(c)
            else:
                try:
                    if isinstance(cols, slice):
                        required.extend(list(df_example.columns[cols]))
                    elif isinstance(cols, str):
                        required.append(cols)
                except Exception:
                    pass
    except Exception:
        pass
    # dedupe & cleanup
    required = [str(c).strip() for c in dict.fromkeys(required) if isinstance(c, str)]
    return required

if model is not None:
    required_input_cols = introspect_required_cols_from_pipeline(model, df)
    if required_input_cols:
        st.info("Model expects additional columns; adding them to manual UI if missing: " + ", ".join(required_input_cols[:10]))
        for col in required_input_cols:
            if col not in X_df.columns:
                # pull sensible default from df if available
                if col in df.columns:
                    try:
                        if np.issubdtype(df[col].dtype, np.number):
                            default = float(df[col].median()) if not df[col].dropna().empty else 0.0
                        else:
                            default = str(df[col].mode().iloc[0]) if not df[col].dropna().empty else ""
                    except Exception:
                        default = 0.0 if col.lower().endswith('_count') else ""
                else:
                    default = "" if not col.lower().endswith('_count') else 0.0
                X_df[col] = default

# special-case: ensure 'platform' exists and is visible at top
if 'platform' not in X_df.columns:
    if 'platform' in df.columns:
        try:
            default_platform = str(df['platform'].mode().iloc[0])
        except Exception:
            default_platform = 'unknown'
    else:
        default_platform = 'unknown'
    X_df['platform'] = default_platform
    # move platform to front
    cols = X_df.columns.tolist()
    cols.remove('platform')
    cols.insert(0, 'platform')
    X_df = X_df[cols]

# recompute feature list after possibly injecting required cols
feature_columns = X_df.columns.tolist()
st.write("Features used for manual input (auto-derived + model-required):")
st.write(feature_columns)

# -------------------------
# Helper: feature alignment and defaults
# -------------------------
def normalize_cols_df(df_local):
    df_local = df_local.copy()
    df_local.columns = [str(c).strip().replace(' ', '_') for c in df_local.columns]
    return df_local

def get_pipeline_input_cols(pipeline, df_example):
    req = []
    try:
        pre = None
        if hasattr(pipeline, 'named_steps') and 'pre' in pipeline.named_steps:
            pre = pipeline.named_steps['pre']
        else:
            if hasattr(pipeline, 'steps'):
                for n, step in pipeline.steps:
                    if isinstance(step, ColumnTransformer):
                        pre = step
                        break
        if pre is None:
            return req
        for name, transformer, cols in pre.transformers_:
            if isinstance(cols, (list, tuple, np.ndarray)):
                for c in cols:
                    if isinstance(c, str):
                        req.append(str(c).strip().replace(' ', '_'))
            else:
                try:
                    if isinstance(cols, slice):
                        req.extend([str(c).strip().replace(' ', '_') for c in list(df_example.columns[cols])])
                    elif isinstance(cols, str):
                        req.append(str(cols).strip().replace(' ', '_'))
                except Exception:
                    pass
    except Exception:
        pass
    return list(dict.fromkeys(req))

def ensure_and_align_input(input_df_local, dataset_df_local, model_pipeline):
    # normalize dataset names
    dataset_df_local = normalize_cols_df(dataset_df_local)
    input_df_local = normalize_cols_df(input_df_local)

    numeric_cols = dataset_df_local.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target]
    categorical_cols = dataset_df_local.select_dtypes(include=['object','category','bool']).columns.tolist()
    if target in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != target]

    expected = []
    if model_pipeline is not None:
        expected = get_pipeline_input_cols(model_pipeline, dataset_df_local)

    if not expected:
        expected = [c for c in dataset_df_local.columns if c != target]

    # fill missing
    for col in expected:
        if col not in input_df_local.columns:
            if col in dataset_df_local.columns and np.issubdtype(dataset_df_local[col].dtype, np.number):
                default_val = float(dataset_df_local[col].median()) if not dataset_df_local[col].dropna().empty else 0.0
            elif col in dataset_df_local.columns:
                try:
                    default_val = str(dataset_df_local[col].mode().iloc[0])
                except Exception:
                    default_val = ""
            else:
                default_val = "unknown" if col.lower()=="platform" else ("" if not col.lower().endswith('_count') else 0.0)
            input_df_local[col] = default_val

    ordered = [c for c in expected if c in input_df_local.columns]
    remaining = [c for c in input_df_local.columns if c not in ordered]
    final_df_local = input_df_local[ordered + remaining].copy()
    return final_df_local, expected

# -------------------------
# Social Media Username Auto-Fetch
# -------------------------
st.markdown("---")
st.subheader("Quick Check - Enter Username")
st.markdown("Enter a social media username to automatically fetch account data and analyze the account.")

st.info("**Note:** This is an AI prediction model and may not always be 100% accurate. Results should be used as guidance, not definitive proof. Usernames with numbers or certain patterns may be incorrectly flagged.")

with st.form("username_form"):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        platform_choice = st.selectbox(
            "Platform",
            ["Twitter/X", "Instagram"],
            help="Select the social media platform"
        )
    with col2:
        social_username = st.text_input(
            "Username",
            placeholder="username (without @)",
            help="Enter username without @ symbol"
        )
    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        fetch_button = st.form_submit_button("Analyze Account", use_container_width=True)

if fetch_button and social_username:
    platform_icon = "🐦" if platform_choice == "Twitter/X" else "📸"
    with st.spinner(f"Fetching {platform_choice} profile data for @{social_username}..."):
        # Fetch profile based on platform
        if platform_choice == "Twitter/X":
            profile_data = fetch_twitter_profile(social_username)
            extract_func = extract_twitter_features
        else:  # Instagram
            profile_data = fetch_instagram_profile(social_username)
            extract_func = extract_instagram_features
        
        if "error" in profile_data:
            st.error(f"{profile_data['error']}")
            st.info("**Tip**: Make sure the username is correct and the account is public. You can also use Manual Input below.")
        else:
            # Extract features
            features = extract_func(profile_data)
            
            # Display fetched data
            st.success(f"Successfully fetched {platform_choice} profile for @{social_username}")
            
            with st.expander("View Profile Data", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Followers", f"{features['followers']:,}")
                col2.metric("Following", f"{features['following']:,}")
                col3.metric("Posts", f"{features['posts']:,}")
                col4.metric("Verified", "Yes" if features['verified'] else "No")
            
            # Prepare for prediction
            input_df_social = pd.DataFrame([features])
            input_df_social = normalize_cols_df(input_df_social)
            
            # Load model if needed
            if model is None:
                model_temp = load_model(MODEL_PATH)
                if model_temp is None:
                    st.error(f"No trained model found at {MODEL_PATH}. Run the pipeline first.")
                    st.stop()
            else:
                model_temp = model
            
            # Align features
            final_df_social, expected_list = ensure_and_align_input(input_df_social.copy(), df, model_temp)
            
            # Predict
            try:
                pred = model_temp.predict(final_df_social)[0]
                prob = model_temp.predict_proba(final_df_social)[0]
                prob_fake = float(prob[1]) if len(prob) >= 2 else float(prob[0])
                
                label_map = {1: "Fake", 0: "Real"}
                display_label = label_map.get(int(pred), str(pred))
                
                # Display result
                st.markdown("---")
                st.subheader("Analysis Result")
                
                if display_label == "Fake":
                    st.warning(f"### This account appears to be **{display_label}**")
                    st.write(f"**Confidence:** {prob_fake*100:.1f}% likely to be fake")
                else:
                    st.info(f"### This account appears to be **{display_label}**")
                    st.write(f"**Confidence:** {(1-prob_fake)*100:.1f}% likely to be real")
                
                with st.expander("View Detailed Analysis"):
                    st.json({
                        "platform": platform_choice,
                        "username": f"@{social_username}",
                        "prediction": display_label,
                        "confidence": f"{max(prob_fake, 1-prob_fake)*100:.2f}%",
                        "fake_probability": f"{prob_fake:.4f}",
                        "real_probability": f"{1-prob_fake:.4f}"
                    })
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------
# Manual input UI form
# -------------------------
st.markdown("---")
with st.expander("Manual Input (Advanced)", expanded=False):
    st.markdown("Enter account features manually for other platforms or custom analysis.")
    with st.form("manual_input_form"):
        user_values = {}
        col1, col2 = st.columns(2)
        for i, col in enumerate(feature_columns):
            series = X_df[col].dropna()
            dtype = X_df[col].dtype
            # numeric
            if np.issubdtype(dtype, np.number):
                try:
                    default_val = float(series.median()) if not series.empty else 0.0
                except Exception:
                    default_val = 0.0
                if i % 2 == 0:
                    user_values[col] = col1.number_input(label=col, value=default_val, step=1.0, format="%.6g")
                else:
                    user_values[col] = col2.number_input(label=col, value=default_val, step=1.0, format="%.6g")
            else:
                uniques = series.unique()[:50].tolist()
                if 1 < len(uniques) <= 20:
                    if i % 2 == 0:
                        user_values[col] = col1.selectbox(label=col, options=uniques, index=0)
                    else:
                        user_values[col] = col2.selectbox(label=col, options=uniques, index=0)
                else:
                    if i % 2 == 0:
                        user_values[col] = col1.text_input(label=col, value=str(series.mode()[0]) if not series.empty else "")
                    else:
                        user_values[col] = col2.text_input(label=col, value=str(series.mode()[0]) if not series.empty else "")

        submitted = st.form_submit_button("Predict")

if not submitted:
    st.info("Fill inputs and press Predict to get a classification.")
    st.stop()

# build input df
input_df = pd.DataFrame([user_values], columns=feature_columns)

# debug & normalize + auto-fill BEFORE prediction
# normalize column names everywhere
df = normalize_cols_df(df)
X_df = normalize_cols_df(X_df)
input_df = normalize_cols_df(input_df)

# get pipeline expected original input columns and show debug info
required_cols = get_pipeline_input_cols(model, df) if model is not None else []
st.write("🔎 Pipeline expects these original input columns (sample):", required_cols[:30])
st.write("🔎 Your input columns:", list(input_df.columns))
missing = set(required_cols) - set(input_df.columns)
st.write("🔎 Missing columns (exact):", sorted(list(missing)))

# auto-fill missing
for col in sorted(list(missing)):
    if col in df.columns and np.issubdtype(df[col].dtype, np.number):
        default_val = float(df[col].median()) if not df[col].dropna().empty else 0.0
    elif col in df.columns:
        try:
            default_val = str(df[col].mode().iloc[0])
        except Exception:
            default_val = ""
    else:
        default_val = "unknown" if col.lower()=="platform" else ""
    input_df[col] = default_val
    st.info(f"Auto-filled missing column '{col}' with default: {default_val}")

# prepare final_df to send to model
final_df, expected_list = ensure_and_align_input(input_df.copy(), df, model)
st.write("🔎 Final columns sent to pipeline (sample):", list(final_df.columns)[:30])

# reload model if needed
if model is None:
    model = load_model(MODEL_PATH)
    if model is None:
        st.error(f"No trained model found at {MODEL_PATH}. Run the pipeline first (python fake_account_pipeline.py).")
        st.stop()

# predict
try:
    pred = model.predict(final_df)[0]
except Exception as e:
    st.error(f"Prediction failed — check that your input column names match the training features. Error: {e}")
    st.stop()

prob_val = None
try:
    prob = model.predict_proba(final_df)[0]
    if len(prob) >= 2:
        prob_val = float(prob[1])
    else:
        prob_val = float(prob[0])
except Exception:
    prob_val = None

label_map = {1: "Fake", 0: "Real"}
display_label = label_map.get(int(pred), str(pred))

st.markdown("---")
st.subheader("Analysis Result")

if display_label == "Fake":
    st.warning(f"### This account appears to be **{display_label}**")
    if prob_val is not None:
        st.write(f"**Confidence:** {prob_val*100:.1f}% likely to be fake")
else:
    st.info(f"### This account appears to be **{display_label}**")
    if prob_val is not None:
        st.write(f"**Confidence:** {(1-prob_val)*100:.1f}% likely to be real")

if prob_val is None:
    st.caption("Note: Probability score not available for this prediction")

with st.expander("View Technical Details"):
    st.json({
        "predicted_label": int(pred), 
        "predicted_text": display_label, 
        "predicted_probability": None if prob_val is None else float(prob_val)
    })

with st.expander("Show input row used for prediction"):
    st.dataframe(final_df.T)

st.caption("If prediction fails, confirm that the pipeline used the same feature names and that a model is saved at ./outputs/best_model.joblib")

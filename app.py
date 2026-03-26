import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# --- Constants for File Management ---
MODEL_FILE = "house_model.pkl"
ENCODERS_FILE = "house_label_encoders.pkl"
SCALER_FILE = "house_scaler.pkl"
FEATURE_NAMES_FILE = "house_feature_names.pkl"

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stSelectbox, .stNumberInput {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 5px;
        padding: 5px;
    }
    .stSelectbox:hover, .stNumberInput:hover {
        border-color: #007bff;
    }
    .prediction-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        font-size: 18px;
        font-weight: bold;
        color: #155724;
    }
    .section-header {
        color: #343a40;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
        border-radius: 10px;
    }
    .stDataFrame {
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to preprocess data and train Random Forest model
@st.cache_resource
def train_model(data_path="data.csv"):
    """Trains the Random Forest model and saves all necessary artifacts."""
    data = pd.read_csv(data_path)

    # Numerical columns
    numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                      'floors', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

    # Fill missing numerical values with median
    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical variables
    categorical_cols = ['city', 'statezip', 'street', 'country', 'waterfront', 'view', 'condition']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Ensure all training data is converted to string for consistent encoding
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Features and target
    X = data.drop(['price', 'date'], axis=1)
    y = data['price']
    
    # --- FIX: Capture the exact feature order used during training ---
    feature_names = X.columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (This code runs if Streamlit is active, which it is, so we remove the redundant/deprecated check)
    y_pred = model.predict(X_test)
    with st.expander("View Model Performance"):
        st.write(f"**Mean Absolute Error (MAE):** ₹{mean_absolute_error(y_test, y_pred):,.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):,.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")

    # Save model artifacts
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_FILE, "wb") as f:
        pickle.dump(label_encoders, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_NAMES_FILE, "wb") as f:
        pickle.dump(feature_names, f)

    return model, label_encoders, scaler, data, feature_names

# Load or train model and data
if not os.path.exists(MODEL_FILE):
    st.info("Model not found. Training model for the first time...")
    model, label_encoders, scaler, data, feature_names = train_model()
    st.success("Model trained and saved!")
else:
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(ENCODERS_FILE, "rb") as f:
            label_encoders = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        # --- FIX: Load the feature names list to ensure correct column order ---
        with open(FEATURE_NAMES_FILE, "rb") as f:
            feature_names = pickle.load(f)
        
        data = pd.read_csv("data.csv")
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}. Retraining the model.")
        model, label_encoders, scaler, data, feature_names = train_model()


# Header
st.markdown("<h1 style='text-align: center; color: #343a40;'>🏠 House Price Prediction & Search</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6c757d;'>Estimate your dream home value or search houses by budget!</h3>", unsafe_allow_html=True)

# Sidebar
page = st.sidebar.selectbox("📋 Choose a Page", ["Predict House Price", "Search Houses by Price Range"])

if page == "Predict House Price":
    st.markdown("<div class='section-header'>Predict House Price</div>", unsafe_allow_html=True)

    # Inputs
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, step=1, value=3)
    bathrooms = st.number_input("🚿 Bathrooms", min_value=0.0, step=0.5, value=2.0)
    sqft_living = st.number_input("📏 Living Area (sqft)", min_value=0, step=100, value=1500)
    sqft_lot = st.number_input("🌳 Lot Size (sqft)", min_value=0, step=100, value=5000)
    floors = st.number_input("🏢 Floors", min_value=0.0, step=0.5, value=1.0)
    sqft_above = st.number_input("⬆️ Sqft Above", min_value=0, step=100, value=1500)
    sqft_basement = st.number_input("⬇️ Sqft Basement", min_value=0, step=100, value=0)
    yr_built = st.number_input("📅 Year Built", min_value=1800, max_value=2025, step=1, value=2000)
    yr_renovated = st.number_input("🔨 Year Renovated", min_value=0, max_value=2025, step=1, value=0)
    
    # Categorical Inputs
    # Ensure options are string converted as done during training
    city = st.selectbox("📍 City", label_encoders['city'].classes_)
    statezip = st.selectbox("🏷️ State/Zip", label_encoders['statezip'].classes_)
    condition = st.selectbox("🏚️ Condition", label_encoders['condition'].classes_)
    waterfront = st.selectbox("🌊 Waterfront", label_encoders['waterfront'].classes_)
    view = st.selectbox("👀 View", label_encoders['view'].classes_)
    street = st.selectbox("🛣️ Street", label_encoders['street'].classes_)
    country = st.selectbox("🌎 Country", label_encoders['country'].classes_)

    if st.button("🔍 Predict Price"):
        try:
            input_data = {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft_living': sqft_living,
                'sqft_lot': sqft_lot,
                'floors': floors,
                'sqft_above': sqft_above,
                'sqft_basement': sqft_basement,
                'yr_built': yr_built,
                'yr_renovated': yr_renovated,
                # Transform categorical inputs using loaded encoders
                'city': label_encoders['city'].transform([str(city)])[0],
                'statezip': label_encoders['statezip'].transform([str(statezip)])[0],
                'condition': label_encoders['condition'].transform([str(condition)])[0],
                'waterfront': label_encoders['waterfront'].transform([str(waterfront)])[0],
                'view': label_encoders['view'].transform([str(view)])[0],
                'street': label_encoders['street'].transform([str(street)])[0],
                'country': label_encoders['country'].transform([str(country)])[0]
            }

            input_df = pd.DataFrame([input_data])

            # --- CRITICAL FIX: Reorder columns to match the training data feature order ---
            input_df = input_df.reindex(columns=feature_names)
            
            # Check for any missing values introduced by bad reindexing
            if input_df.isnull().values.any():
                st.error("Internal Error: Feature mapping failed. Please report this issue.")
            else:
                # Scale numerical features using the loaded scaler
                numerical_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                                'floors', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
                input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

                prediction = model.predict(input_df)
                st.markdown(f"<div class='prediction-box'>✅ Predicted House Price: ₹{prediction[0]:,.2f}</div>", unsafe_allow_html=True)

        except ValueError as e:
             st.error(f"Prediction Error: The model could not process the inputs. This often happens if a new categorical value is selected that was not present in the training data. Details: {e}")
        except Exception as e:
             st.error(f"An unexpected error occurred during prediction: {e}")


elif page == "Search Houses by Price Range":
    st.markdown("<div class='section-header'>Search Houses by Price Range</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("💰 Minimum Price (₹)", min_value=0, step=50000, value=500000)
    with col2:
        max_price = st.number_input("💰 Maximum Price (₹)", min_value=0, step=50000, value=1000000)

    if st.button("🔎 Search"):
        if max_price < min_price:
            st.error("🚫 Maximum price must be greater than or equal to minimum price.")
        else:
            filtered_data = data[(data['price'] >= min_price) & (data['price'] <= max_price)]

            if not filtered_data.empty:
                filtered_data = filtered_data.reset_index(drop=True)
                filtered_data.insert(0, 'Serial No.', range(1, len(filtered_data) + 1))

                # Decode the categorical columns for user display (optional but helpful)
                display_data = filtered_data.copy()
                for col, le in label_encoders.items():
                    # Check if the column exists and needs decoding (was encoded during training)
                    if col in display_data.columns and display_data[col].dtype != object:
                         try:
                             # Inverse transform the encoded integer column back to original strings
                             display_data[col] = le.inverse_transform(display_data[col].astype(int))
                         except:
                             # Handle cases where inverse_transform might fail (e.g., if the column was not actually encoded)
                             pass

                display_cols = ['Serial No.', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
                                'sqft_lot', 'floors', 'yr_built', 'yr_renovated', 'city', 'statezip', 'condition']
                st.markdown("**Houses within the specified price range:**")
                st.dataframe(display_data[display_cols], use_container_width=True)
            else:
                st.warning("⚠️ No houses found in the specified price range.")
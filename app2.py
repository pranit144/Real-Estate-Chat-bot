from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import re
from typing import Dict, List, Any, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import sklearn # Import sklearn to get version

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAU8NFxxscrLcPcYqp5bnnyBPTVH0v1aZY') # IMPORTANT: Replace with a secure way to manage API keys
genai.configure(api_key=GEMINI_API_KEY)
# Using gemini-1.5-flash for potentially larger context window and better performance
model = genai.GenerativeModel('gemini-2.0-flash')


class EnhancedRealEstateSalesAssistant:
    def __init__(self):
        self.pipeline = None  # Initialize pipeline to None
        self.training_feature_names = [] # To store feature names from training
        self.model_version = 'N/A'
        self.load_model_and_data()
        self.conversation_history = []
        self.client_preferences = {}
        self.last_search_results = None
        self.context_memory = []

    def _clean_gemini_text(self, text: str) -> str:
        """Removes common Markdown formatting characters (like asterisks, hashes) from Gemini's output."""
        if not isinstance(text, str):
            return text # Return as is if not a string

        # Remove bold/italic markers (**)
        cleaned_text = re.sub(r'\*\*', '', text)
        # Remove italic markers (*)
        cleaned_text = re.sub(r'\*', '', cleaned_text)
        # Remove header markers (#) at the start of a line, and any following whitespace
        cleaned_text = re.sub(r'^\s*#+\s*', '', cleaned_text, flags=re.MULTILINE)
        # Remove typical list bullet points like '- ' or '+ ' or '* ' at the start of a line
        # (Note: '*' removal above might catch list items, this is for redundancy/clarity)
        cleaned_text = re.sub(r'^\s*[-+]\s+', '', cleaned_text, flags=re.MULTILINE)
        # Replace multiple newlines with a single one (or two, depending on desired paragraph spacing)
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        # Remove excessive leading/trailing whitespace from each line
        cleaned_text = '\n'.join([line.strip() for line in cleaned_text.splitlines()])
        # Final strip for the whole block
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    def load_model_and_data(self):
        """Load the trained model and dataset with version compatibility"""
        try:
            # Load the dataset first
            self.df = pd.read_csv('Pune_House_Data.csv')

            # Clean and process data
            self.df = self.clean_dataset()

            # Extract unique values for filtering
            self.locations = sorted(self.df['site_location'].unique().tolist())
            self.area_types = sorted(self.df['area_type'].unique().tolist())
            self.availability_types = sorted(self.df['availability'].unique().tolist())
            self.bhk_options = sorted(self.df['bhk'].unique().tolist())

            # Try to load model with version compatibility
            model_loaded = self.load_model_with_version_check()

            if not model_loaded:
                logger.warning("Model loading failed, prediction will use fallback method.")
                self.pipeline = None # Ensure pipeline is None if loading failed

            logger.info("✅ Model and data loading process completed.")
            logger.info(f"Dataset shape: {self.df.shape}")
            logger.info(f"Available locations: {len(self.locations)}")

        except Exception as e:
            logger.error(f"❌ Critical error loading model/data: {e}")
            # Create fallback data
            self.df = self.create_fallback_data()
            self.locations = ['Baner', 'Hadapsar', 'Kothrud', 'Viman Nagar', 'Wakad', 'Hinjewadi']
            self.area_types = ['Built-up Area', 'Super built-up Area', 'Plot Area', 'Carpet Area']
            self.availability_types = ['Ready To Move', 'Not Ready', 'Under Construction']
            self.bhk_options = [1, 2, 3, 4, 5, 6]
            self.pipeline = None # Ensure pipeline is None if data loading failed

    def load_model_with_version_check(self):
        """Load model with version compatibility check, or retrain if necessary."""
        try:
            # First try to load the new version-compatible model
            try:
                model_data = joblib.load('house_price_prediction_v2.pkl')
                self.pipeline = model_data['pipeline']
                self.training_feature_names = model_data.get('feature_names', [])
                self.model_version = model_data.get('version', '2.0')
                logger.info(f"✅ Loaded model version {self.model_version} with {len(self.training_feature_names)} features.")
                return True
            except FileNotFoundError:
                logger.info("New model version (house_price_prediction_v2.pkl) not found, attempting to retrain.")
                return self.retrain_model_with_current_version()
            except Exception as v2_load_error:
                logger.error(f"Failed to load new model version (v2): {v2_load_error}. Attempting retraining.")
                return self.retrain_model_with_current_version()

        except Exception as e:
            logger.error(f"Error in model loading process: {e}. Retraining will be attempted as a last resort.")
            return self.retrain_model_with_current_version() # Final attempt to retrain if other loading fails

    def retrain_model_with_current_version(self):
        """Retrain the model with current scikit-learn version"""
        try:
            logger.info(f"🔄 Retraining model with scikit-learn version {sklearn.__version__}")

            # Prepare features (this method also sets self.training_feature_names)
            X, y = self.create_training_data_with_proper_features()

            if X is None or y is None:
                logger.error("Failed to create training data, cannot retrain model.")
                self.pipeline = None
                return False

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                ))
            ])

            # Train model
            logger.info("Training model...")
            pipeline.fit(X_train, y_train)

            # Test model
            y_pred = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logger.info(f"✅ Model trained successfully:")
            logger.info(f"   MAE: {mae:.2f} lakhs")
            logger.info(f"   R² Score: {r2:.3f}")

            # Save the new model
            self.pipeline = pipeline
            # self.training_feature_names is already set by create_training_data_with_proper_features

            model_data = {
                'pipeline': self.pipeline,
                'feature_names': self.training_feature_names,
                'version': '2.0',
                'sklearn_version': sklearn.__version__,
                'training_info': {
                    'mae': mae,
                    'r2_score': r2,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': len(self.training_feature_names)
                },
                'created_at': datetime.now().isoformat()
            }

            # Save with version info
            joblib.dump(model_data, 'house_price_prediction_v2.pkl')
            logger.info("✅ Model saved as house_price_prediction_v2.pkl")

            return True

        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            self.pipeline = None
            return False

    def create_training_data_with_proper_features(self):
        """Create training data with proper feature engineering for consistent predictions"""
        try:
            # Prepare features for training
            df_features = self.df.copy()

            # Select relevant columns for training
            feature_columns = ['site_location', 'area_type', 'availability', 'bhk', 'bath', 'balcony', 'total_sqft']

            # Check if all required columns exist
            missing_columns = [col for col in feature_columns + ['price'] if col not in df_features.columns]
            if missing_columns:
                logger.error(f"Missing essential columns for training: {missing_columns}")
                return None, None

            df_features = df_features[feature_columns + ['price']]

            # Clean data (ensure numeric columns are correct, fillna where appropriate)
            df_features['bhk'] = pd.to_numeric(df_features['bhk'], errors='coerce')
            df_features['bath'] = pd.to_numeric(df_features['bath'], errors='coerce').fillna(df_features['bath'].median())
            df_features['balcony'] = pd.to_numeric(df_features['balcony'], errors='coerce').fillna(df_features['balcony'].median())
            df_features['total_sqft'] = pd.to_numeric(df_features['total_sqft'], errors='coerce')
            df_features['price'] = pd.to_numeric(df_features['price'], errors='coerce')

            # Remove rows with critical missing values for training
            df_features = df_features.dropna(subset=['price', 'total_sqft', 'bhk', 'site_location'])

            # Remove outliers before one-hot encoding for more robust training
            df_features = self.remove_outliers(df_features)

            # One-hot encode categorical variables
            categorical_cols = ['site_location', 'area_type', 'availability']
            df_encoded = pd.get_dummies(df_features, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)

            # Separate features and target
            X = df_encoded.drop('price', axis=1)
            y = df_encoded['price']

            # Store feature names for later use in prediction
            self.training_feature_names = X.columns.tolist()

            logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")

            return X, y

        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return None, None

    def remove_outliers(self, df):
        """Remove outliers from the dataset based on price and area."""
        try:
            # Calculate price_per_sqft for outlier removal, then drop it if not needed as a feature
            df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']

            # Remove price outliers (e.g., beyond 3 standard deviations from mean)
            price_mean = df['price'].mean()
            price_std = df['price'].std()
            df = df[abs(df['price'] - price_mean) <= 3 * price_std]

            # Remove area outliers (e.g., beyond 3 standard deviations from mean)
            area_mean = df['total_sqft'].mean()
            area_std = df['total_sqft'].std()
            df = df[abs(df['total_sqft'] - area_mean) <= 3 * area_std]

            # Remove properties with unrealistic price per sqft (e.g., <1000 or >20000 INR/sqft for Pune)
            df = df[(df['price_per_sqft'] >= 1000) & (df['price_per_sqft'] <= 20000)]

            # Drop temporary price_per_sqft column
            df = df.drop(columns=['price_per_sqft'])

            logger.info(f"Dataset after outlier removal: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return df # Return original df if error occurs

    def predict_single_price(self, location, bhk, bath, balcony, sqft, area_type, availability):
        """Predict price for a single property using ML model, with fallback to heuristic."""
        try:
            # Check if ML pipeline is loaded
            if not hasattr(self, 'pipeline') or self.pipeline is None or not self.training_feature_names:
                logger.warning("ML pipeline or training features not available. Using fallback prediction.")
                return self.predict_single_price_fallback(location, bhk, bath, balcony, sqft, area_type, availability)

            try:
                # Create a feature dictionary initialized with 0s for all expected features
                feature_dict = {feature: 0 for feature in self.training_feature_names}

                # Set numerical features
                numerical_mapping = {
                    'bhk': bhk,
                    'bath': bath,
                    'balcony': balcony,
                    'total_sqft': sqft
                }
                for feature, value in numerical_mapping.items():
                    if feature in feature_dict:
                        feature_dict[feature] = value
                    else:
                        logger.warning(f"Numerical feature '{feature}' not found in training features. Skipping.")

                # Set categorical features (one-hot encoded)
                # Location
                location_key = f"site_location_{location}"
                if location_key in feature_dict:
                    feature_dict[location_key] = 1
                else:
                    found_loc = False
                    for feature in self.training_feature_names:
                        if feature.startswith('site_location_') and location.lower() in feature.lower():
                            feature_dict[feature] = 1
                            found_loc = True
                            break
                    if not found_loc:
                        logger.warning(f"Location '{location}' not found as exact or fuzzy match in training features. This location will not contribute to the ML prediction specific to its one-hot encoding.")

                # Area Type
                area_type_key = f"area_type_{area_type}"
                if area_type_key in feature_dict:
                    feature_dict[area_type_key] = 1
                else:
                    found_area_type = False
                    for feature in self.training_feature_names:
                        if feature.startswith('area_type_') and area_type.lower() in feature.lower():
                            feature_dict[feature] = 1
                            found_area_type = True
                            break
                    if not found_area_type:
                        logger.warning(f"Area type '{area_type}' not found in training features. This area type will not contribute to the ML prediction specific to its one-hot encoding.")

                # Availability
                availability_key = f"availability_{availability}"
                if availability_key in feature_dict:
                    feature_dict[availability_key] = 1
                else:
                    found_availability = False
                    for feature in self.training_feature_names:
                        if feature.startswith('availability_') and availability.lower() in feature.lower():
                            feature_dict[feature] = 1
                            found_availability = True
                            break
                    if not found_availability:
                        logger.warning(f"Availability '{availability}' not found in training features. This availability will not contribute to the ML prediction specific to its one-hot encoding.")

                # Create DataFrame from the feature dictionary and ensure column order
                input_df = pd.DataFrame([feature_dict])[self.training_feature_names]

                predicted_price = self.pipeline.predict(input_df)[0]
                return round(predicted_price, 2)

            except Exception as ml_prediction_error:
                logger.warning(f"ML prediction failed for input {location, bhk, sqft}: {ml_prediction_error}. Falling back to heuristic model.")
                return self.predict_single_price_fallback(location, bhk, bath, balcony, sqft, area_type, availability)

        except Exception as general_error:
            logger.error(f"General error in predict_single_price wrapper: {general_error}. Falling back to heuristic.")
            return self.predict_single_price_fallback(location, bhk, bath, balcony, sqft, area_type, availability)


    def predict_single_price_fallback(self, location, bhk, bath, balcony, sqft, area_type, availability):
        """Enhanced fallback prediction method when ML model is unavailable or fails."""
        try:
            # Base price per sqft for different areas in Pune (extensive map for better fallback)
            location_price_map = {
                'Koregaon Park': 12000, 'Kalyani Nagar': 11000, 'Boat Club Road': 10500,
                'Aundh': 9500, 'Baner': 8500, 'Balewadi': 8000, 'Kothrud': 8500,
                'Viman Nagar': 7500, 'Wakad': 7000, 'Hinjewadi': 7500, 'Pune': 7000,
                'Hadapsar': 6500, 'Kharadi': 7200, 'Yerawada': 7000, 'Lohegaon': 6000,
                'Wagholi': 5500, 'Mundhwa': 7000, 'Undri': 6500, 'Kondhwa': 6000,
                'Katraj': 5800, 'Dhankawadi': 6000, 'Warje': 6500, 'Karve Nagar': 7500,
                'Bavdhan': 8000, 'Pashan': 7500, 'Sus': 7200, 'Pimpri': 6000,
                'Chinchwad': 6200, 'Akurdi': 5800, 'Nigdi': 6500, 'Bhosari': 5500,
                'Chakan': 4800, 'Talegaon': 4500, 'Alandi': 4200, 'Dehu': 4000,
                'Lonavala': 5000, 'Kamshet': 4500, 'Sinhagad Road': 6800,
                'Balaji Nagar': 5800, 'Parvati': 5500, 'Gultekdi': 5200, 'Wanowrie': 6200,
                'Shivajinagar': 9000, 'Deccan': 8500, 'Camp': 7500, 'Koregaon': 8000,
                'Sadashiv Peth': 8200, 'Shukrawar Peth': 7800, 'Kasba Peth': 7500,
                'Narayan Peth': 7200, 'Rasta Peth': 7000, 'Ganj Peth': 6800,
                'Magarpatta': 7500, 'Fursungi': 5000, 'Handewadi': 4800,
                'Mahatma Phule Peth': 6200, 'Bhavani Peth': 6000, 'Bibwewadi': 7000
            }

            # Get base price per sqft, try fuzzy matching if exact match not found
            base_price_per_sqft = location_price_map.get(location, 6500) # Default if no match
            if location not in location_price_map:
                for loc_key, price in location_price_map.items():
                    if location.lower() in loc_key.lower() or loc_key.lower() in location.lower():
                        base_price_per_sqft = price
                        break

            # Area type multiplier
            area_type_multiplier = {
                'Super built-up Area': 1.0,
                'Built-up Area': 0.92, # Typically slightly less than super built-up for same listed price
                'Plot Area': 0.85,    # Plot area might translate to lower built-up
                'Carpet Area': 1.08   # Carpet area is typically more expensive per sqft
            }.get(area_type, 1.0)

            # Availability multiplier (Under Construction/Not Ready typically cheaper)
            availability_multiplier = {
                'Ready To Move': 1.0,
                'Under Construction': 0.88,
                'Not Ready': 0.82
            }.get(availability, 0.95)

            # BHK-based pricing adjustments (larger BHKs might have slightly lower price/sqft due to bulk)
            bhk_multiplier = {
                1: 1.15,  # Premium for 1BHK
                2: 1.0,   # Base
                3: 0.95,  # Slight economy
                4: 0.92,
                5: 0.90,
                6: 0.88
            }.get(bhk, 1.0)

            # Bathroom and Balcony multipliers (simple linear adjustment)
            bath_multiplier = 1.0 + (bath - 2) * 0.03 # Each extra bath adds 3% to price
            balcony_multiplier = 1.0 + (balcony - 1) * 0.02 # Each extra balcony adds 2% to price

            # Ensure multipliers don't go negative or too low
            bath_multiplier = max(0.9, bath_multiplier)
            balcony_multiplier = max(0.95, balcony_multiplier)

            # Calculate final estimated price
            estimated_price_raw = (sqft * base_price_per_sqft *
                                   area_type_multiplier * availability_multiplier *
                                   bhk_multiplier * bath_multiplier * balcony_multiplier)

            estimated_price_lakhs = estimated_price_raw / 100000

            return round(estimated_price_lakhs, 2)

        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
            return None # Indicate failure

    def clean_dataset(self):
        """Clean and process the dataset"""
        df = self.df.copy()

        # Rename 'size' to 'bhk' based on previous logic (if 'size' exists)
        if 'size' in df.columns and 'bhk' not in df.columns:
            df['bhk'] = df['size'].str.extract(r'(\d+)').astype(float)
        elif 'bhk' not in df.columns:
            df['bhk'] = np.nan # Ensure bhk column exists even if 'size' is missing

        # Clean total_sqft column
        df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')

        # Clean price column
        df['price'] = pd.to_numeric(df['price'], errors='coerce')

        # Fill missing values for numerical columns (using median for robustness)
        df['bath'] = pd.to_numeric(df['bath'], errors='coerce').fillna(df['bath'].median())
        df['balcony'] = pd.to_numeric(df['balcony'], errors='coerce').fillna(df['balcony'].median())

        # Fill missing values for categorical columns
        df['society'] = df['society'].fillna('Not Specified')
        df['area_type'] = df['area_type'].fillna('Super built-up Area') # Common default
        df['availability'] = df['availability'].fillna('Ready To Move') # Common default
        df['site_location'] = df['site_location'].fillna('Pune') # General default

        # Remove rows with critical missing values after initial cleaning
        df = df.dropna(subset=['price', 'total_sqft', 'bhk', 'site_location'])

        # Ensure 'bhk' is an integer (or float if decimals are possible)
        df['bhk'] = df['bhk'].astype(int)
        df['bath'] = df['bath'].astype(int)
        df['balcony'] = df['balcony'].astype(int)

        logger.info(f"Dataset cleaned. Original rows: {self.df.shape[0]}, Cleaned rows: {df.shape[0]}")
        return df

    def create_fallback_data(self):
        """Create fallback data if CSV loading fails"""
        logger.warning("Creating fallback dataset due to CSV loading failure.")
        fallback_data = {
            'area_type': ['Super built-up Area', 'Built-up Area', 'Carpet Area'] * 10,
            'availability': ['Ready To Move', 'Under Construction', 'Ready To Move'] * 10,
            'size': ['2 BHK', '3 BHK', '4 BHK'] * 10, # Keep 'size' for cleaning process if needed
            'society': ['Sample Society'] * 30,
            'total_sqft': [1000, 1200, 1500] * 10,
            'bath': [2, 3, 4] * 10,
            'balcony': [1, 2, 3] * 10,
            'price': [50, 75, 100] * 10,
            'site_location': ['Baner', 'Hadapsar', 'Kothrud'] * 10,
        }
        df_fallback = pd.DataFrame(fallback_data)
        # Ensure 'bhk' is generated if 'size' is used, and other derived columns
        if 'size' in df_fallback.columns:
            df_fallback['bhk'] = df_fallback['size'].str.extract(r'(\d+)').astype(float)
        else:
            df_fallback['bhk'] = [2, 3, 4] * 10 # Direct bhk if 'size' is not used
        # Add a placeholder for price_per_sqft as it's used in analysis/outlier removal
        df_fallback['price_per_sqft'] = (df_fallback['price'] * 100000) / df_fallback['total_sqft']
        return df_fallback


    def filter_properties(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Filter properties based on user criteria"""
        filtered_df = self.df.copy()

        # Apply filters
        if filters.get('location'):
            # Use a more robust search for location, allowing partial or case-insensitive matches
            search_location = filters['location'].lower()
            filtered_df = filtered_df[filtered_df['site_location'].str.lower().str.contains(search_location, na=False)]

        if filters.get('bhk') is not None: # Check for None explicitly as bhk can be 0
            filtered_df = filtered_df[filtered_df['bhk'] == filters['bhk']]

        if filters.get('min_price'):
            filtered_df = filtered_df[filtered_df['price'] >= filters['min_price']]

        if filters.get('max_price'):
            filtered_df = filtered_df[filtered_df['price'] <= filters['max_price']]

        if filters.get('min_area'):
            filtered_df = filtered_df[filtered_df['total_sqft'] >= filters['min_area']]

        if filters.get('max_area'):
            filtered_df = filtered_df[filtered_df['total_sqft'] <= filters['max_area']]

        if filters.get('area_type'):
            search_area_type = filters['area_type'].lower()
            filtered_df = filtered_df[filtered_df['area_type'].str.lower().str.contains(search_area_type, na=False)]

        if filters.get('availability'):
            search_availability = filters['availability'].lower()
            filtered_df = filtered_df[filtered_df['availability'].str.lower().str.contains(search_availability, na=False)]

        return filtered_df.head(20)  # Limit results for display

    def get_price_range_analysis(self, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price range and provide insights"""
        if filtered_df.empty:
            return {
                "total_properties": 0,
                "price_range": {"min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0},
                "area_range": {"min": 0.0, "max": 0.0, "avg": 0.0},
                "price_per_sqft": {"min": 0.0, "max": 0.0, "avg": 0.0},
                "location_distribution": {},
                "area_type_distribution": {},
                "availability_distribution": {}
            }

        analysis = {
            "total_properties": len(filtered_df),
            "price_range": {
                "min": float(filtered_df['price'].min()),
                "max": float(filtered_df['price'].max()),
                "avg": float(filtered_df['price'].mean()),
                "median": float(filtered_df['price'].median())
            },
            "area_range": {
                "min": float(filtered_df['total_sqft'].min()),
                "max": float(filtered_df['total_sqft'].max()),
                "avg": float(filtered_df['total_sqft'].mean())
            },
            # Recalculate price_per_sqft for analysis as it might be dropped after outlier removal
            "price_per_sqft": {
                "min": float(((filtered_df['price'] * 100000) / filtered_df['total_sqft']).min()),
                "max": float(((filtered_df['price'] * 100000) / filtered_df['total_sqft']).max()),
                "avg": float(((filtered_df['price'] * 100000) / filtered_df['total_sqft']).mean())
            },
            "location_distribution": filtered_df['site_location'].value_counts().to_dict(),
            "area_type_distribution": filtered_df['area_type'].value_counts().to_dict(),
            "availability_distribution": filtered_df['availability'].value_counts().to_dict()
        }

        return analysis

    def predict_prices_for_filtered_results(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Predict prices for all filtered properties and add to DataFrame."""
        if filtered_df.empty:
            return filtered_df

        predictions = []
        for index, row in filtered_df.iterrows():
            predicted_price = self.predict_single_price(
                location=row['site_location'],
                bhk=row['bhk'],
                bath=row['bath'],
                balcony=row['balcony'],
                sqft=row['total_sqft'],
                area_type=row['area_type'],
                availability=row['availability']
            )
            # Use original price if prediction fails (returns None)
            predictions.append(predicted_price if predicted_price is not None else row['price'])

        filtered_df['predicted_price'] = predictions
        filtered_df['price_difference'] = filtered_df['predicted_price'] - filtered_df['price']
        # Avoid division by zero if original price is 0
        filtered_df['price_difference_pct'] = (
            filtered_df['price_difference'] / filtered_df['price'].replace(0, np.nan) * 100
        ).fillna(0) # Fill NaN with 0 if original price was 0 or prediction failed

        return filtered_df

    def generate_deep_insights(self, filtered_df: pd.DataFrame, analysis: Dict[str, Any]) -> str:
        """Generate deep insights using Gemini AI"""
        try:
            # Prepare context for Gemini
            properties_sample_str = filtered_df[['site_location', 'bhk', 'total_sqft', 'price', 'predicted_price', 'price_difference_pct']].head().to_string(index=False)
            if properties_sample_str:
                properties_sample_str = f"Top 5 Properties Sample:\n{properties_sample_str}\n"
            else:
                properties_sample_str = "No specific property samples available to list."


            context = f"""
            You are a highly experienced real estate market expert specializing in Pune properties.
            Your task is to analyze the provided property data and offer comprehensive, actionable insights for a potential property buyer.

            Here's a summary of the current search results:
            - Total Properties Found: {analysis['total_properties']}
            - Price Range: ₹{analysis['price_range']['min']:.2f}L to ₹{analysis['price_range']['max']:.2f}L
            - Average Price: ₹{analysis['price_range']['avg']:.2f}L
            - Median Price: ₹{analysis['price_range']['median']:.2f}L
            - Average Area: {analysis['area_range']['avg']:.0f} sq ft
            - Average Price per sq ft: ₹{analysis['price_per_sqft']['avg']:.0f}

            Location Distribution: {json.dumps(analysis['location_distribution'], indent=2)}
            Area Type Distribution: {json.dumps(analysis['area_type_distribution'], indent=2)}
            Availability Distribution: {json.dumps(analysis['availability_distribution'], indent=2)}

            {properties_sample_str}

            Please provide detailed market insights including:
            1.  **Current Market Trends**: What are the prevailing trends in Pune's real estate market based on this data? (e.g., buyer's/seller's market, growth areas, demand patterns).
            2.  **Price Competitiveness**: How do the prices of these properties compare to the overall Pune market? Are they underpriced, overpriced, or fair value? Highlight any anomalies based on the 'price_difference_pct' if available.
            3.  **Best Value Propositions**: Identify what types of properties (BHK, area, location) offer the best value for money right now based on average price per sqft and price ranges.
            4.  **Location-Specific Insights**: For the top 2-3 locations, provide specific observations on amenities, connectivity, future development potential, and why they might be good/bad for investment or living.
            5.  **Investment Recommendations**: Based on the data, what would be your general investment advice for someone looking into Pune real estate?
            6.  **Future Market Outlook**: Briefly discuss the short-to-medium term outlook (next 1-2 years) for the Pune real estate market.

            Ensure your response is conversational, professional, and directly actionable for a property buyer.
            """

            response = model.generate_content(context)
            return self._clean_gemini_text(response.text) # Apply cleaning here

        except Exception as e:
            logger.error(f"Error generating insights using Gemini: {e}")
            return "I'm analyzing the market data to provide you with detailed insights. Based on the current search results, I can see various opportunities in your preferred areas. Please bear with me while I gather more information."

    def get_nearby_properties_gemini(self, location: str, requirements: Dict[str, Any]) -> str:
        """Get nearby properties and recent market data using Gemini"""
        try:
            # Refine requirements to be more human-readable for Gemini
            req_str = ""
            for k, v in requirements.items():
                if k == 'min_price': req_str += f"Minimum Budget: {v} Lakhs. "
                elif k == 'max_price': req_str += f"Maximum Budget: {v} Lakhs. "
                elif k == 'min_area': req_str += f"Minimum Area: {v} sq ft. "
                elif k == 'max_area': req_str += f"Maximum Area: {v} sq ft. "
                elif k == 'bhk': req_str += f"BHK: {v}. "
                elif k == 'area_type': req_str += f"Area Type: {v}. "
                elif k == 'availability': req_str += f"Availability: {v}. "
                elif k == 'location': req_str += f"Preferred Location: {v}. "
            if not req_str: req_str = "No specific requirements provided beyond location."


            context = f"""
            You are a knowledgeable real estate advisor with up-to-date market information for Pune.

            A client is interested in properties near **{location}**.
            Their current general requirements are: {req_str.strip()}

            Please provide a concise but informative overview focusing on:
            1.  **Recent Property Launches**: Mention any significant residential projects launched in {location} or very close vicinity in the last 6-12 months.
            2.  **Upcoming Projects**: Are there any major upcoming residential or commercial developments (e.g., IT parks, malls, new infrastructure) that could impact property values in {location} or adjacent areas?
            3.  **Infrastructure Developments**: Discuss any planned or ongoing infrastructure improvements (roads, metro, civic amenities) that might affect property values and lifestyle in {location}.
            4.  **Market Trends in {location}**: What are the current buying/rental trends specific to {location}? Is it a high-demand area, or is supply outpacing demand?
            5.  **Comparative Analysis**: Briefly compare {location} with one or two similar, nearby localities in terms of property values, amenities, and lifestyle.
            6.  **Investment Potential**: What is the general investment outlook and potential rental yield for properties in {location}?

            Focus on specific details relevant to the Pune real estate market in 2024-2025.
            """

            response = model.generate_content(context)
            return self._clean_gemini_text(response.text) # Apply cleaning here

        except Exception as e:
            logger.error(f"Error getting nearby properties from Gemini: {e}")
            return f"I'm gathering information about recent properties and market trends near {location}. This area has shown consistent growth in property values, and I'll get you more specific details shortly."

    def update_context_memory(self, user_query: str, results: Dict[str, Any]):
        """Update context memory to maintain conversation continuity"""
        self.context_memory.append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'results_summary': {
                'total_properties': results.get('total_properties', 0),
                'price_range': results.get('price_range', {}),
                'locations': list(results.get('location_distribution', {}).keys())
            }
        })

        # Keep only last 5 interactions
        if len(self.context_memory) > 5:
            self.context_memory.pop(0)

    def extract_requirements(self, user_message: str) -> Dict[str, Any]:
        """Extract property requirements from user message"""
        requirements = {}
        message_lower = user_message.lower()

        # Extract BHK
        bhk_match = re.search(r'(\d+)\s*bhk', message_lower)
        if bhk_match:
            try:
                requirements['bhk'] = int(bhk_match.group(1))
            except ValueError:
                pass # Ignore if not a valid number

        # Extract budget range (flexible with "lakh", "crore", "million")
        budget_match = re.search(
            r'(?:budget|price|cost)(?:.*?(?:of|between|from)\s*)?'
            r'(\d+(?:\.\d+)?)\s*(?:lakhs?|crores?|million)?(?:(?:\s*to|\s*-\s*)\s*(\d+(?:\.\d+)?)\s*(?:lakhs?|crores?|million)?)?',
            message_lower
        )
        if budget_match:
            try:
                min_amount = float(budget_match.group(1))
                max_amount = float(budget_match.group(2)) if budget_match.group(2) else None

                # Detect units for both min and max if provided
                if 'crore' in budget_match.group(0): # Check original matched string for "crore"
                    requirements['min_price'] = min_amount * 100
                    if max_amount:
                        requirements['max_price'] = max_amount * 100
                elif 'million' in budget_match.group(0):
                    requirements['min_price'] = min_amount * 10
                    if max_amount:
                        requirements['max_price'] = max_amount * 10
                else: # Default to lakhs
                    requirements['min_price'] = min_amount
                    if max_amount:
                        requirements['max_price'] = max_amount
            except ValueError:
                pass

        # Extract location
        found_location = False
        for location in self.locations:
            if location.lower() in message_lower:
                requirements['location'] = location
                found_location = True
                break
        # Broader location matching if specific location not found
        if not found_location:
            general_locations = ['pune', 'pcmp'] # Add more general terms if needed
            for gen_loc in general_locations:
                if gen_loc in message_lower:
                    requirements['location'] = gen_loc.capitalize() # Or a default location like 'Pune'
                    break


        # Extract area requirements
        area_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:to|-)?\s*(\d+(?:\.\d+)?)?\s*(?:sq\s*ft|sqft|square\s*feet|sft)',
                               message_lower)
        if area_match:
            try:
                requirements['min_area'] = float(area_match.group(1))
                if area_match.group(2):
                    requirements['max_area'] = float(area_match.group(2))
            except ValueError:
                pass

        # Extract area type
        for area_type in self.area_types:
            if area_type.lower() in message_lower:
                requirements['area_type'] = area_type
                break

        # Extract availability
        for availability in self.availability_types:
            if availability.lower() in message_lower:
                requirements['availability'] = availability
                break

        return requirements

    def handle_query(self, user_message: str) -> Dict[str, Any]:
        """Main function to handle user queries with context awareness"""
        try:
            # Extract requirements from current message
            requirements = self.extract_requirements(user_message)

            # Merge with previous context if building on last query
            if self.is_follow_up_query(user_message):
                requirements = self.merge_with_context(requirements)

            # Update client preferences (store all extracted preferences)
            self.client_preferences.update(requirements)

            # Filter properties based on requirements
            filtered_df = self.filter_properties(requirements)

            # If no properties found, provide suggestions
            if filtered_df.empty:
                return self.handle_no_results(requirements)

            # Predict prices for filtered results
            filtered_df = self.predict_prices_for_filtered_results(filtered_df)

            # Get price range analysis
            analysis = self.get_price_range_analysis(filtered_df)

            # Generate deep insights
            insights = self.generate_deep_insights(filtered_df, analysis)

            # Get nearby properties information
            nearby_info = ""
            if requirements.get('location'):
                nearby_info = self.get_nearby_properties_gemini(requirements['location'], requirements)

            # Prepare response
            response_data = {
                'filtered_properties': filtered_df.to_dict('records'),
                'analysis': analysis,
                'insights': insights,
                'nearby_properties': nearby_info,
                'requirements': requirements,
                'context_aware': self.is_follow_up_query(user_message)
            }

            # Update context memory
            self.update_context_memory(user_message, analysis)
            self.last_search_results = response_data # Store the full response

            return response_data

        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return {'error': f'An error occurred while processing your query: {str(e)}'}

    def is_follow_up_query(self, user_message: str) -> bool:
        """Check if this is a follow-up query building on previous results"""
        follow_up_indicators = ['also', 'and', 'additionally', 'what about', 'show me more', 'similar', 'nearby',
                                'around', 'close to', 'next', 'other', 'different', 'change', 'update']
        return any(indicator in user_message.lower() for indicator in follow_up_indicators) and len(
            self.context_memory) > 0

    def merge_with_context(self, new_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new requirements with context from previous queries"""
        # Start with the overall client preferences
        merged = self.client_preferences.copy()

        # Overwrite with any new requirements from the current message
        merged.update(new_requirements)

        # Apply default or previously established values if not specified in current query
        # Example: If a new query doesn't specify location, use the last known location
        if 'location' not in new_requirements and 'location' in self.client_preferences:
            merged['location'] = self.client_preferences['location']

        # You can add more sophisticated merging logic here if needed
        # e.g., if a new budget is provided, replace the old one entirely.
        # The current update() method already handles this for direct key conflicts.

        return merged

    def handle_no_results(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cases where no properties match the criteria"""
        suggestions = ["Try relaxing some of your criteria."]

        if requirements.get('max_price'):
            suggestions.append(f"Consider increasing your budget above ₹{requirements['max_price']:.2f}L.")
            # Suggest a slightly higher range
            suggestions.append(f"Perhaps a range of ₹{requirements['max_price']:.2f}L to ₹{(requirements['max_price'] * 1.2):.2f}L?")

        if requirements.get('location'):
            # Find nearby popular locations in the dataset, excluding the one searched
            nearby_locs = [loc for loc in self.locations if loc.lower() != requirements['location'].lower()]
            if nearby_locs:
                # Sort by count if possible, or just pick top 3
                top_locations = self.df['site_location'].value_counts().index.tolist()
                suggested_nearby = [loc for loc in top_locations if loc.lower() != requirements['location'].lower()][:3]
                if suggested_nearby:
                    suggestions.append(f"Consider nearby popular areas like: {', '.join(suggested_nearby)}.")

        if requirements.get('bhk') is not None:
            # Suggest +/- 1 BHK if not already at min/max reasonable bhk
            if requirements['bhk'] > 1:
                suggestions.append(f"Consider looking for {requirements['bhk'] - 1} BHK options.")
            if requirements['bhk'] < 6: # Assuming 6 is a practical upper limit for most searches
                suggestions.append(f"Consider {requirements['bhk'] + 1} BHK options.")

        if requirements.get('max_area'):
            suggestions.append(f"You might find more options if you consider properties up to ~{(requirements['max_area'] * 1.1):.0f} sq ft.")

        # Default message if specific suggestions are not generated
        if len(suggestions) == 1 and "relaxing" in suggestions[0]:
            final_suggestion_text = "No properties found matching your exact criteria. Please try broadening your search or clarifying your preferences."
        else:
            final_suggestion_text = f"No properties found matching your exact criteria. Here are some suggestions: {'; '.join(suggestions)}."

        return {
            'filtered_properties': [],
            'analysis': self.get_price_range_analysis(pd.DataFrame()), # Empty analysis
            'insights': final_suggestion_text,
            'nearby_properties': '',
            'requirements': requirements,
            'suggestions': suggestions
        }


# Initialize the enhanced sales assistant (MUST be after class definition)
sales_assistant = EnhancedRealEstateSalesAssistant()


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with comprehensive property search"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Please provide your query'}), 400

        # Get comprehensive response from sales assistant
        response_data = sales_assistant.handle_query(user_message)

        # Ensure response_data always includes necessary keys even on error
        if 'error' in response_data:
            return jsonify({
                'response': response_data,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'conversation_context': len(sales_assistant.context_memory)
            }), 500 # Return 500 for internal errors
        else:
            return jsonify({
                'response': response_data,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'conversation_context': len(sales_assistant.context_memory)
            })

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({'error': f'An unexpected error occurred in chat endpoint: {str(e)}'}), 500


# Flask routes must be defined at the module level, not inside a class
@app.route('/model_status')
def model_status():
    """Get current model status and version info"""
    try:
        status = {
            'model_loaded': hasattr(sales_assistant, 'pipeline') and sales_assistant.pipeline is not None,
            'model_version': getattr(sales_assistant, 'model_version', 'unknown'),
            'feature_count': len(getattr(sales_assistant, 'training_feature_names', [])),
            'prediction_method': 'ML' if hasattr(sales_assistant, 'pipeline') and sales_assistant.pipeline else 'Fallback',
            'sklearn_version': sklearn.__version__
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in model_status endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Retrain the model with current version"""
    try:
        logger.info("Starting model retraining via endpoint...")
        success = sales_assistant.retrain_model_with_current_version()

        if success:
            return jsonify({
                'message': 'Model retrained successfully',
                'version': sales_assistant.model_version,
                'feature_count': len(sales_assistant.training_feature_names),
                'sample_features': sales_assistant.training_feature_names[:10] if sales_assistant.training_feature_names else []
            })
        else:
            return jsonify({'error': 'Failed to retrain model. Check logs for details.'}), 500

    except Exception as e:
        logger.error(f"Retrain endpoint error: {e}")
        return jsonify({'error': f'An unexpected error occurred during retraining: {str(e)}'}), 500


@app.route('/filter_properties', methods=['POST'])
def filter_properties_endpoint(): # Renamed to avoid conflict with class method
    """Dedicated endpoint for property filtering"""
    try:
        filters = request.get_json()

        filtered_df = sales_assistant.filter_properties(filters)
        filtered_df = sales_assistant.predict_prices_for_filtered_results(filtered_df)
        analysis = sales_assistant.get_price_range_analysis(filtered_df)

        return jsonify({
            'properties': filtered_df.to_dict('records'),
            'analysis': analysis,
            'total_count': len(filtered_df)
        })

    except Exception as e:
        logger.error(f"Filter endpoint error: {e}")
        return jsonify({'error': f'Filtering error: {str(e)}'}), 500


@app.route('/get_insights', methods=['POST'])
def get_insights():
    """Get AI-powered insights for current search results"""
    try:
        data = request.get_json()
        location_from_request = data.get('location')
        requirements_from_request = data.get('requirements', {})

        if sales_assistant.last_search_results:
            filtered_df = pd.DataFrame(sales_assistant.last_search_results['filtered_properties'])
            analysis = sales_assistant.last_search_results['analysis']
            insights = sales_assistant.generate_deep_insights(filtered_df, analysis)

            nearby_info = ""
            # Use location from request if provided, otherwise from last search results
            effective_location = location_from_request or sales_assistant.last_search_results['requirements'].get('location')
            if effective_location:
                nearby_info = sales_assistant.get_nearby_properties_gemini(effective_location, requirements_from_request or sales_assistant.last_search_results['requirements'])

            return jsonify({
                'insights': insights,
                'nearby_properties': nearby_info
            })
        else:
            return jsonify({'error': 'No previous search results available for insights. Please perform a search first.'}), 400

    except Exception as e:
        logger.error(f"Insights endpoint error: {e}")
        return jsonify({'error': f'Insights error: {str(e)}'}), 500


@app.route('/get_market_data')
def get_market_data():
    """Get comprehensive market data"""
    try:
        # Ensure df is loaded before accessing its properties
        if not hasattr(sales_assistant, 'df') or sales_assistant.df.empty:
            return jsonify({'error': 'Market data not available, dataset not loaded.'}), 500

        market_data = {
            'locations': sales_assistant.locations,
            'area_types': sales_assistant.area_types,
            'availability_types': sales_assistant.availability_types,
            'bhk_options': sales_assistant.bhk_options,
            'price_statistics': {
                'min_price': float(sales_assistant.df['price'].min()),
                'max_price': float(sales_assistant.df['price'].max()),
                'avg_price': float(sales_assistant.df['price'].mean()),
                'median_price': float(sales_assistant.df['price'].median())
            },
            'area_statistics': {
                'min_area': float(sales_assistant.df['total_sqft'].min()),
                'max_area': float(sales_assistant.df['total_sqft'].max()),
                'avg_area': float(sales_assistant.df['total_sqft'].mean())
            }
        }

        return jsonify(market_data)

    except Exception as e:
        logger.error(f"Market data endpoint error: {e}")
        return jsonify({'error': f'Market data error: {str(e)}'}), 500


@app.route('/client_preferences')
def get_client_preferences():
    """Get current client preferences and conversation history"""
    return jsonify({
        'preferences': sales_assistant.client_preferences,
        'conversation_history': sales_assistant.context_memory,
        'last_search_available': sales_assistant.last_search_results is not None
    })


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset entire session including preferences and context"""
    try:
        sales_assistant.client_preferences = {}
        sales_assistant.context_memory = []
        sales_assistant.last_search_results = None
        return jsonify({'message': 'Session reset successfully'})
    except Exception as e:
        logger.error(f"Reset session error: {e}")
        return jsonify({'error': f'Failed to reset session: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': hasattr(sales_assistant, 'pipeline') and sales_assistant.pipeline is not None,
            'model_version': sales_assistant.model_version,
            'data_loaded': hasattr(sales_assistant, 'df') and not sales_assistant.df.empty,
            'dataset_size': len(sales_assistant.df) if hasattr(sales_assistant, 'df') else 0,
            'locations_available': len(sales_assistant.locations),
            'context_memory_size': len(sales_assistant.context_memory),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


if __name__ == '__main__':
    # It's better to explicitly set the host for deployment environments.
    # For development, debug=True is useful but should be False in production.
    app.run(debug=True, host='0.0.0.0', port=5000)
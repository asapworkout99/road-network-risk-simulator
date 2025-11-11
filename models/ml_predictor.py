"""
ML Model Predictor
Supports CatBoost, LightGBM, and TabNet models
Loads trained models and makes risk predictions
"""

import joblib
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, Union, List
import streamlit as st
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class RiskPredictor:
    """
    Load and use trained ML model for accident risk prediction
    Supports multiple model types: CatBoost, LightGBM, TabNet
    """
    
    def __init__(self, model_path: str, preprocessor_path: str, config_path: str):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model file (.joblib or .pkl)
            preprocessor_path: Path to preprocessor file (.pkl)
            config_path: Path to model config JSON
        """
        self.model = None
        self.preprocessor = None
        self.config = None
        self.model_type = None
        
        try:
            # Load model configuration
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                    self.model_type = self.config.get('model_type', 'unknown')
            
            # Load model
            if model_path.endswith('.joblib'):
                self.model = joblib.load(model_path)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            # Load preprocessor
            if os.path.exists(preprocessor_path):
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
            
            print(f"✅ Loaded {self.model_type} model successfully")
            
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def preprocess_features(self, features: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input features using fitted preprocessor
        
        Args:
            features: Dict or DataFrame with 12 input features
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        # Convert dict to DataFrame
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        df = features.copy()
        
        # Ensure all required features are present
        required_features = self.config.get('feature_names', Config.FEATURE_NAMES)
        for feat in required_features:
            if feat not in df.columns:
                # Set default values for missing features
                if feat in Config.CATEGORICAL_FEATURES:
                    if feat == 'road_type':
                        df[feat] = 'urban'
                    elif feat == 'lighting':
                        df[feat] = 'daylight'
                    elif feat == 'weather':
                        df[feat] = 'clear'
                    elif feat == 'time_of_day':
                        df[feat] = 'afternoon'
                    else:
                        df[feat] = 'unknown'
                elif feat in Config.BOOLEAN_FEATURES:
                    df[feat] = False
                else:
                    df[feat] = 0
        
        # Apply preprocessor if available
        if self.preprocessor:
            # Get components from preprocessor package
            scaler = self.preprocessor.get('scaler')
            label_encoders = self.preprocessor.get('label_encoders', {})
            categorical_features = self.preprocessor.get('categorical_features', Config.CATEGORICAL_FEATURES)
            numerical_features = self.preprocessor.get('numerical_features', Config.NUMERICAL_FEATURES)
            boolean_features = self.preprocessor.get('boolean_features', Config.BOOLEAN_FEATURES)
            
            # Encode categorical features
            for col in categorical_features:
                if col in df.columns and col in label_encoders:
                    try:
                        df[col] = label_encoders[col].transform(df[col].astype(str))
                    except:
                        # Handle unknown categories
                        df[col] = 0
            
            # Convert boolean to int
            for col in boolean_features:
                if col in df.columns:
                    df[col] = df[col].astype(int)
            
            # Scale numerical features
            if scaler and numerical_features:
                num_cols = [col for col in numerical_features if col in df.columns]
                if num_cols:
                    df[num_cols] = scaler.transform(df[num_cols])
        
        return df[required_features]
    
    def predict(self, features: Union[Dict, pd.DataFrame]) -> float:
        """
        Predict accident risk for a single road segment
        
        Args:
            features: Dict or DataFrame with 12 input features
            
        Returns:
            float: Predicted risk score (0-1)
        """
        try:
            # Preprocess
            X = self.preprocess_features(features)
            
            # Predict
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(X)[0]
            else:
                raise AttributeError("Model does not have predict method")
            
            # Clip to valid range [0, 1]
            return float(np.clip(prediction, 0.0, 1.0))
            
        except Exception as e:
            st.warning(f"Prediction failed: {str(e)}. Using fallback calculation.")
            # Fallback to simplified risk calculation
            return self._fallback_risk_calculation(features)
    
    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict risk for multiple road segments
        
        Args:
            features_df: DataFrame with 12 input features per row
            
        Returns:
            Array of predicted risk scores (0-1)
        """
        try:
            X = self.preprocess_features(features_df)
            predictions = self.model.predict(X)
            return np.clip(predictions, 0.0, 1.0)
        except Exception as e:
            st.warning(f"Batch prediction failed: {str(e)}")
            # Return default moderate risk for each row
            return np.full(len(features_df), 0.5)
    
    def _fallback_risk_calculation(self, features: Union[Dict, pd.DataFrame]) -> float:
        """
        Simplified risk calculation when model fails
        Uses heuristic rules based on feature values
        """
        if isinstance(features, pd.DataFrame):
            features = features.iloc[0].to_dict()
        
        risk = 0.0
        
        # Curvature risk (30% weight)
        risk += features.get('curvature', 0.3) * 0.3
        
        # Speed limit risk (25% weight)
        speed = features.get('speed_limit', 50)
        normalized_speed = (speed - 25) / (120 - 25)  # Normalize to 0-1
        risk += max(0, min(normalized_speed, 1)) * 0.25
        
        # Weather risk (15% weight)
        weather_risk = {
            'clear': 0.0,
            'rainy': 0.10,
            'foggy': 0.15
        }
        risk += weather_risk.get(features.get('weather', 'clear'), 0)
        
        # Lighting risk (15% weight)
        lighting_risk = {
            'daylight': 0.0,
            'dim': 0.08,
            'night': 0.15
        }
        risk += lighting_risk.get(features.get('lighting', 'daylight'), 0)
        
        # Road type risk (10% weight)
        road_risk = {
            'urban': 0.0,
            'rural': 0.05,
            'highway': 0.10
        }
        risk += road_risk.get(features.get('road_type', 'urban'), 0)
        
        # Accident history risk (capped at 15%)
        accidents = features.get('num_reported_accidents', 0)
        risk += min(accidents * 0.03, 0.15)
        
        # No road signs penalty (5%)
        if not features.get('road_signs_present', True):
            risk += 0.05
        
        # Holiday/school season (minor adjustments)
        if features.get('holiday', False):
            risk += 0.02
        if features.get('school_season', False):
            risk += 0.02
        
        # Time of day
        time_risk = {
            'morning': 0.02,  # Rush hour
            'afternoon': 0.0,
            'evening': 0.03   # Rush hour + fatigue
        }
        risk += time_risk.get(features.get('time_of_day', 'afternoon'), 0)
        
        # Ensure risk is between 0 and 1
        return min(max(risk, 0.0), 1.0)
    
    def get_model_info(self) -> Dict:
        """Get model information and performance metrics"""
        info = {
            'model_type': self.model_type,
            'config': self.config,
            'has_preprocessor': self.preprocessor is not None
        }
        
        # Try to load training results
        if os.path.exists(Config.TRAINING_RESULTS_PATH):
            try:
                with open(Config.TRAINING_RESULTS_PATH, 'r') as f:
                    info['training_results'] = json.load(f)
            except:
                pass
        
        # Add model attributes if available
        if hasattr(self.model, 'feature_names_'):
            info['feature_names'] = self.model.feature_names_
        
        if hasattr(self.model, 'n_features_'):
            info['n_features'] = self.model.n_features_
        
        return info
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from model"""
        
        # Try to load from saved file first
        if os.path.exists(Config.FEATURE_IMPORTANCE_PATH):
            try:
                with open(Config.FEATURE_IMPORTANCE_PATH, 'r') as f:
                    data = json.load(f)
                    return pd.DataFrame({
                        'feature': data['features'],
                        'importance': data['importance']
                    }).sort_values('importance', ascending=False)
            except:
                pass
        
        # Try to get from model directly
        if hasattr(self.model, 'feature_importances_'):
            features = self.config.get('feature_names', Config.FEATURE_NAMES)
            return pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Try for CatBoost
        if hasattr(self.model, 'get_feature_importance'):
            try:
                features = self.config.get('feature_names', Config.FEATURE_NAMES)
                importance = self.model.get_feature_importance()
                return pd.DataFrame({
                    'feature': features,
                    'importance': importance
                }).sort_values('importance', ascending=False)
            except:
                pass
        
        # Return empty DataFrame if no importance available
        return pd.DataFrame()
    
    def explain_prediction(self, features: Dict) -> Dict:
        """
        Explain a prediction by showing feature contributions
        
        Args:
            features: Feature dictionary
            
        Returns:
            Dictionary with explanation
        """
        risk_score = self.predict(features)
        
        explanation = {
            'risk_score': risk_score,
            'risk_level': Config.get_risk_level(risk_score),
            'feature_values': features,
            'risk_factors': []
        }
        
        # Identify high-risk factors
        if features.get('curvature', 0) > 0.6:
            explanation['risk_factors'].append('High road curvature')
        
        if features.get('speed_limit', 50) > 80:
            explanation['risk_factors'].append('High speed limit')
        
        if features.get('weather') in ['rainy', 'foggy']:
            explanation['risk_factors'].append(f"Poor weather: {features.get('weather')}")
        
        if features.get('lighting') == 'night':
            explanation['risk_factors'].append('Night driving')
        
        if features.get('num_reported_accidents', 0) > 2:
            explanation['risk_factors'].append('High accident history')
        
        if not features.get('road_signs_present', True):
            explanation['risk_factors'].append('Limited road signage')
        
        return explanation


@st.cache_resource
def load_risk_predictor():
    """
    Load ML risk predictor (cached for performance)
    Falls back to simplified calculation if model not found
    
    Returns:
        RiskPredictor instance or None if files not found
    """
    try:
        # Check if model files exist
        model_exists = os.path.exists(Config.MODEL_PATH)
        preprocessor_exists = os.path.exists(Config.PREPROCESSOR_PATH)
        config_exists = os.path.exists(Config.MODEL_CONFIG_PATH)
        
        if model_exists and preprocessor_exists and config_exists:
            predictor = RiskPredictor(
                Config.MODEL_PATH,
                Config.PREPROCESSOR_PATH,
                Config.MODEL_CONFIG_PATH
            )
            st.success("✅ ML model loaded successfully!")
            return predictor
        else:
            st.warning("⚠️ ML model files not found. Using simplified risk calculation.")
            
            # Show which files are missing
            missing_files = []
            if not model_exists:
                missing_files.append("best_model.joblib")
            if not preprocessor_exists:
                missing_files.append("preprocessor.pkl")
            if not config_exists:
                missing_files.append("model_config.json")
            
            st.info(f"📁 Missing files in models/ directory:\n- " + "\n- ".join(missing_files))
            st.info("💡 The app will use fallback risk calculations until you add your trained model.")
            
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Using fallback risk calculation.")
        return None


def create_sample_model_config():
    """
    Create a sample model_config.json file for reference
    Users should customize this for their actual model
    """
    sample_config = {
        "model_type": "CatBoost",
        "version": "1.0",
        "training_date": "2024-01-15",
        "feature_names": Config.FEATURE_NAMES,
        "categorical_features": Config.CATEGORICAL_FEATURES,
        "numerical_features": Config.NUMERICAL_FEATURES,
        "boolean_features": Config.BOOLEAN_FEATURES,
        "target_variable": "risk_score",
        "description": "Accident risk prediction model",
        "metrics": {
            "mse": 0.0234,
            "mae": 0.1123,
            "r2": 0.8567
        }
    }
    
    # Save to models directory
    config_path = os.path.join(Config.MODELS_DIR, 'model_config_SAMPLE.json')
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    return sample_config


# Utility function for testing
def test_predictor():
    """Test the predictor with sample data"""
    print("🧪 Testing Risk Predictor...")
    
    # Sample features
    test_features = {
        'road_type': 'highway',
        'num_lanes': 2,
        'curvature': 0.7,
        'speed_limit': 90,
        'lighting': 'night',
        'weather': 'rainy',
        'road_signs_present': False,
        'public_road': True,
        'time_of_day': 'evening',
        'holiday': False,
        'school_season': False,
        'num_reported_accidents': 3
    }
    
    # Try to load predictor
    predictor = load_risk_predictor()
    
    if predictor:
        risk = predictor.predict(test_features)
        print(f"✅ Predicted risk: {risk:.4f}")
        
        explanation = predictor.explain_prediction(test_features)
        print(f"Risk level: {explanation['risk_level']}")
        print(f"Risk factors: {explanation['risk_factors']}")
    else:
        print("⚠️ Using fallback calculation")
        predictor_obj = RiskPredictor.__new__(RiskPredictor)
        risk = predictor_obj._fallback_risk_calculation(test_features)
        print(f"Fallback risk: {risk:.4f}")


if __name__ == "__main__":
    test_predictor()

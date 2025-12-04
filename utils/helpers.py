"""
Helper utility functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from config import Config

def format_distance(distance_m: float) -> str:
    """
    Format distance for display
    
    Args:
        distance_m: Distance in meters
        
    Returns:
        Formatted string (e.g., "1.5 km" or "250 m")
    """
    if distance_m >= 1000:
        return f"{distance_m / 1000:.2f} km"
    else:
        return f"{distance_m:.0f} m"


def format_risk_level(risk_score: float) -> Tuple[str, str]:
    """
    Convert risk score to level and color
    
    Args:
        risk_score: Risk score (0-1)
        
    Returns:
        Tuple of (level_name, color)
    """
    if risk_score < Config.RISK_LOW:
        return "Low", "green"
    elif risk_score < Config.RISK_MEDIUM:
        return "Medium", "yellow"
    elif risk_score < Config.RISK_HIGH:
        return "High", "orange"
    else:
        return "Very High", "red"


def calculate_route_time(distance_km: float, avg_speed_kmh: float = 50) -> str:
    """
    Calculate estimated travel time
    
    Args:
        distance_km: Distance in kilometers
        avg_speed_kmh: Average speed in km/h
        
    Returns:
        Formatted time string (e.g., "15 min" or "1h 30min")
    """
    hours = distance_km / avg_speed_kmh
    
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} min"
    else:
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h}h {m}min"


def generate_risk_summary(network) -> Dict:
    """
    Generate comprehensive risk summary for network
    
    Args:
        network: NetworkManager instance
        
    Returns:
        Dictionary with summary statistics
    """
    risks = [seg.risk_score for seg in network.segments.values()]
    
    if not risks:
        return {
            'total_segments': 0,
            'avg_risk': 0,
            'risk_variance': 0,
            'high_risk_percentage': 0,
            'risk_categories': {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        }
    
    # Categorize risks
    low = sum(1 for r in risks if r < Config.RISK_LOW)
    medium = sum(1 for r in risks if Config.RISK_LOW <= r < Config.RISK_MEDIUM)
    high = sum(1 for r in risks if Config.RISK_MEDIUM <= r < Config.RISK_HIGH)
    very_high = sum(1 for r in risks if r >= Config.RISK_HIGH)
    
    return {
        'total_segments': len(risks),
        'avg_risk': np.mean(risks),
        'risk_variance': np.var(risks),
        'risk_std': np.std(risks),
        'min_risk': np.min(risks),
        'max_risk': np.max(risks),
        'median_risk': np.median(risks),
        'high_risk_percentage': (high + very_high) / len(risks) * 100,
        'risk_categories': {
            'low': low,
            'medium': medium,
            'high': high,
            'very_high': very_high
        },
        'risk_percentages': {
            'low': low / len(risks) * 100,
            'medium': medium / len(risks) * 100,
            'high': high / len(risks) * 100,
            'very_high': very_high / len(risks) * 100
        }
    }


def get_risk_recommendations(risk_score: float, segment_features: Dict) -> List[str]:
    """
    Generate safety recommendations based on risk factors
    
    Args:
        risk_score: Predicted risk score
        segment_features: Road segment features
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # General recommendations based on risk level
    if risk_score >= 0.75:
        recommendations.append("⚠️ VERY HIGH RISK: Consider alternative route if possible")
        recommendations.append("🚗 Drive with extreme caution")
        recommendations.append("📱 Share your route with someone")
    elif risk_score >= 0.50:
        recommendations.append("⚠️ HIGH RISK: Exercise extra caution")
        recommendations.append("🚗 Reduce speed and increase following distance")
    elif risk_score >= 0.25:
        recommendations.append("⚡ MODERATE RISK: Stay alert")
    else:
        recommendations.append("✅ LOW RISK: Normal driving conditions")
    
    # Specific recommendations based on features
    if segment_features.get('curvature', 0) > 0.6:
        recommendations.append("🔄 Sharp curves ahead - reduce speed")
    
    if segment_features.get('speed_limit', 50) > 70:
        recommendations.append("⚡ High-speed zone - maintain safe distance")
    
    weather = segment_features.get('weather', 'clear')
    if weather == 'rainy':
        recommendations.append("🌧️ Wet conditions - reduce speed by 10-20%")
    elif weather == 'foggy':
        recommendations.append("🌫️ Low visibility - use fog lights and reduce speed")
    
    lighting = segment_features.get('lighting', 'daylight')
    if lighting == 'night':
        recommendations.append("🌙 Night driving - use high beams when safe")
    elif lighting == 'dim':
        recommendations.append("🌆 Poor lighting - drive defensively")
    
    if not segment_features.get('road_signs_present', True):
        recommendations.append("⚠️ Limited road signage - be extra alert")
    
    accidents = segment_features.get('num_reported_accidents', 0)
    if accidents > 3:
        recommendations.append(f"📊 High accident history ({accidents} reported) - exercise caution")
    
    return recommendations


def export_route_to_csv(route_info: Dict, filename: str = "route_export.csv"):
    """
    Export route information to CSV
    
    Args:
        route_info: Route dictionary from PathFinder
        filename: Output filename
    """
    segments_data = []
    
    for i, segment in enumerate(route_info['segments']):
        segments_data.append({
            'segment_num': i + 1,
            'from_node': segment['from'],
            'to_node': segment['to'],
            'length_m': segment['length'],
            'risk_score': segment['risk']
        })
    
    df = pd.DataFrame(segments_data)
    df.to_csv(filename, index=False)
    
    return df


def calculate_accident_probability(risk_score: float, distance_km: float) -> Dict:
    """
    Estimate accident probability for a route
    
    Args:
        risk_score: Average risk score for route
        distance_km: Route distance in km
        
    Returns:
        Dictionary with probability estimates
    """
    # Base accident rate per km (hypothetical)
    base_rate_per_km = 0.001  # 0.1% per km
    
    # Adjust by risk score
    adjusted_rate = base_rate_per_km * (1 + risk_score * 10)
    
    # Calculate probability for entire route
    route_probability = 1 - (1 - adjusted_rate) ** distance_km
    
    return {
        'probability_percent': route_probability * 100,
        'risk_level': format_risk_level(risk_score)[0],
        'interpretation': _interpret_probability(route_probability)
    }


def _interpret_probability(probability: float) -> str:
    """Interpret accident probability"""
    if probability < 0.001:
        return "Extremely low likelihood"
    elif probability < 0.01:
        return "Very low likelihood"
    elif probability < 0.05:
        return "Low likelihood"
    elif probability < 0.10:
        return "Moderate likelihood"
    else:
        return "Elevated likelihood - exercise caution"


def simulate_condition_changes(segment_features: Dict, predictor) -> pd.DataFrame:
    """
    Simulate how different conditions affect risk
    
    Args:
        segment_features: Base segment features
        predictor: RiskPredictor instance
        
    Returns:
        DataFrame with condition scenarios and risks
    """
    scenarios = []
    
    # Base scenario
    base_features = segment_features.copy()
    if predictor:
        base_risk = predictor.predict(base_features)
    else:
        base_risk = 0.5
    
    scenarios.append({
        'scenario': 'Current',
        'weather': base_features.get('weather'),
        'lighting': base_features.get('lighting'),
        'time': base_features.get('time_of_day'),
        'risk_score': base_risk
    })
    
    # Weather scenarios
    for weather in ['clear', 'rainy', 'foggy']:
        features = base_features.copy()
        features['weather'] = weather
        if predictor:
            risk = predictor.predict(features)
        else:
            risk = base_risk + {'clear': 0, 'rainy': 0.1, 'foggy': 0.15}.get(weather, 0)
        
        scenarios.append({
            'scenario': f'Weather: {weather.capitalize()}',
            'weather': weather,
            'lighting': base_features.get('lighting'),
            'time': base_features.get('time_of_day'),
            'risk_score': min(risk, 1.0)
        })
    
    # Lighting scenarios
    for lighting in ['daylight', 'dim', 'night']:
        features = base_features.copy()
        features['lighting'] = lighting
        if predictor:
            risk = predictor.predict(features)
        else:
            risk = base_risk + {'daylight': 0, 'dim': 0.08, 'night': 0.15}.get(lighting, 0)
        
        scenarios.append({
            'scenario': f'Lighting: {lighting.capitalize()}',
            'weather': base_features.get('weather'),
            'lighting': lighting,
            'time': base_features.get('time_of_day'),
            'risk_score': min(risk, 1.0)
        })
    
    return pd.DataFrame(scenarios)


def validate_model_files() -> Dict[str, bool]:
    """
    Check if all required model files exist
    
    Returns:
        Dictionary with file existence status
    """
    import os
    
    files = {
        'model': Config.MODEL_PATH,
        'preprocessor': Config.PREPROCESSOR_PATH,
        'config': Config.MODEL_CONFIG_PATH,
        'training_results': Config.TRAINING_RESULTS_PATH,
        'feature_importance': Config.FEATURE_IMPORTANCE_PATH
    }
    
    status = {}
    for name, path in files.items():
        status[name] = os.path.exists(path)
    
    return status

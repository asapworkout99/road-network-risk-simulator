"""
Network Manager - Road Network Data Structures
Handles road network creation, manipulation, and analysis
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st

class RoadSegment:
    """Represents a single road segment with its properties"""
    
    def __init__(self, segment_id: str, start_node: str, end_node: str, 
                 length: float, **properties):
        self.id = segment_id
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.properties = properties
        self.risk_score = properties.get('risk_score', 0.0)
    
    def update_risk(self, risk_score: float):
        """Update risk score for this segment"""
        self.risk_score = risk_score
        self.properties['risk_score'] = risk_score
    
    def get_features(self) -> Dict:
        """Get ML features for risk prediction"""
        return {
            'road_type': self.properties.get('road_type', 'urban'),
            'num_lanes': self.properties.get('num_lanes', 2),
            'curvature': self.properties.get('curvature', 0.3),
            'speed_limit': self.properties.get('speed_limit', 50),
            'lighting': self.properties.get('lighting', 'daylight'),
            'weather': self.properties.get('weather', 'clear'),
            'road_signs_present': self.properties.get('road_signs_present', True),
            'public_road': self.properties.get('public_road', True),
            'time_of_day': self.properties.get('time_of_day', 'afternoon'),
            'holiday': self.properties.get('holiday', False),
            'school_season': self.properties.get('school_season', False),
            'num_reported_accidents': self.properties.get('num_reported_accidents', 0)
        }


class NetworkManager:
    """
    Manages road network graph and operations
    Uses NetworkX for graph operations
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.segments = {}
        self.nodes = {}
    
    def add_node(self, node_id: str, lat: float = None, lon: float = None, **attributes):
        """Add a node (intersection) to the network"""
        self.graph.add_node(node_id, lat=lat, lon=lon, **attributes)
        self.nodes[node_id] = {'lat': lat, 'lon': lon, **attributes}
    
    def add_segment(self, segment: RoadSegment):
        """Add a road segment to the network"""
        self.segments[segment.id] = segment
        
        # Add edge to graph with properties
        self.graph.add_edge(
            segment.start_node,
            segment.end_node,
            key=segment.id,
            length=segment.length,
            risk_score=segment.risk_score,
            **segment.properties
        )
    
    def create_grid_network(self, rows: int = 5, cols: int = 5, 
                           spacing: float = 100.0) -> None:
        """
        Create a simple grid network for testing
        
        Args:
            rows: Number of rows
            cols: Number of columns
            spacing: Distance between nodes (meters)
        """
        self.graph.clear()
        self.segments.clear()
        self.nodes.clear()
        
        # Create nodes
        for i in range(rows):
            for j in range(cols):
                node_id = f"N_{i}_{j}"
                lat = 6.5244 + (i * spacing / 111000)  # Lagos coordinates
                lon = 3.3792 + (j * spacing / 111000)
                self.add_node(node_id, lat=lat, lon=lon)
        
        # Create horizontal segments
        segment_id = 0
        for i in range(rows):
            for j in range(cols - 1):
                start = f"N_{i}_{j}"
                end = f"N_{i}_{j+1}"
                
                segment = RoadSegment(
                    segment_id=f"S_{segment_id}",
                    start_node=start,
                    end_node=end,
                    length=spacing,
                    road_type=np.random.choice(['urban', 'rural', 'highway']),
                    num_lanes=np.random.randint(1, 4),
                    curvature=np.random.uniform(0.1, 0.8),
                    speed_limit=np.random.choice([30, 50, 70, 90]),
                    lighting='daylight',
                    weather='clear',
                    road_signs_present=np.random.choice([True, False]),
                    public_road=True,
                    time_of_day='afternoon',
                    holiday=False,
                    school_season=True,
                    num_reported_accidents=np.random.randint(0, 5)
                )
                self.add_segment(segment)
                segment_id += 1
        
        # Create vertical segments
        for i in range(rows - 1):
            for j in range(cols):
                start = f"N_{i}_{j}"
                end = f"N_{i+1}_{j}"
                
                segment = RoadSegment(
                    segment_id=f"S_{segment_id}",
                    start_node=start,
                    end_node=end,
                    length=spacing,
                    road_type=np.random.choice(['urban', 'rural', 'highway']),
                    num_lanes=np.random.randint(1, 4),
                    curvature=np.random.uniform(0.1, 0.8),
                    speed_limit=np.random.choice([30, 50, 70, 90]),
                    lighting='daylight',
                    weather='clear',
                    road_signs_present=np.random.choice([True, False]),
                    public_road=True,
                    time_of_day='afternoon',
                    holiday=False,
                    school_season=True,
                    num_reported_accidents=np.random.randint(0, 5)
                )
                self.add_segment(segment)
                segment_id += 1
    
    def update_segment_risk(self, segment_id: str, risk_score: float):
        """Update risk score for a segment"""
        if segment_id in self.segments:
            self.segments[segment_id].update_risk(risk_score)
            
            # Update graph edge
            segment = self.segments[segment_id]
            if self.graph.has_edge(segment.start_node, segment.end_node):
                self.graph[segment.start_node][segment.end_node][segment_id]['risk_score'] = risk_score
    
    def get_segment_features_df(self) -> pd.DataFrame:
        """Get all segment features as DataFrame"""
        features_list = []
        for seg_id, segment in self.segments.items():
            features = segment.get_features()
            features['segment_id'] = seg_id
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def update_all_risks(self, risk_predictor):
        """Update risk scores for all segments using ML predictor"""
        if not risk_predictor:
            return
        
        features_df = self.get_segment_features_df()
        segment_ids = features_df['segment_id'].values
        features_df = features_df.drop('segment_id', axis=1)
        
        # Predict risks
        risks = risk_predictor.predict_batch(features_df)
        
        # Update segments
        for seg_id, risk in zip(segment_ids, risks):
            self.update_segment_risk(seg_id, risk)
    
    def get_statistics(self) -> Dict:
        """Get network statistics"""
        risks = [seg.risk_score for seg in self.segments.values()]
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_segments': len(self.segments),
            'total_length_km': sum(seg.length for seg in self.segments.values()) / 1000,
            'avg_risk': np.mean(risks) if risks else 0,
            'max_risk': max(risks) if risks else 0,
            'min_risk': min(risks) if risks else 0,
            'high_risk_segments': sum(1 for r in risks if r > 0.7)
        }
    
    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """Get node positions for visualization"""
        positions = {}
        for node_id, data in self.nodes.items():
            if data['lat'] is not None and data['lon'] is not None:
                positions[node_id] = (data['lon'], data['lat'])
        return positions
    
    def export_to_geojson(self) -> Dict:
        """Export network to GeoJSON format"""
        features = []
        
        for seg_id, segment in self.segments.items():
            start_node = self.nodes[segment.start_node]
            end_node = self.nodes[segment.end_node]
            
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [
                        [start_node['lon'], start_node['lat']],
                        [end_node['lon'], end_node['lat']]
                    ]
                },
                'properties': {
                    'segment_id': seg_id,
                    'risk_score': segment.risk_score,
                    'length': segment.length,
                    **segment.properties
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }

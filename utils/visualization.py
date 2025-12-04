"""
Visualization utilities for road networks and risk analysis
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import folium
from folium import plugins
import streamlit as st
from streamlit_folium import folium_static
from config import Config

class NetworkVisualizer:
    """Create visualizations for road networks"""
    
    @staticmethod
    def plot_network_graph(network, highlighted_path: Optional[List[str]] = None):
        """
        Create interactive network graph with Plotly
        
        Args:
            network: NetworkManager instance
            highlighted_path: List of node IDs to highlight as path
        """
        positions = network.get_node_positions()
        
        if not positions:
            st.warning("No node positions available for visualization")
            return None
        
        # Create edge traces
        edge_traces = []
        
        for seg_id, segment in network.segments.items():
            start_pos = positions.get(segment.start_node)
            end_pos = positions.get(segment.end_node)
            
            if start_pos and end_pos:
                risk = segment.risk_score
                
                # Determine color based on risk
                if risk < Config.RISK_LOW:
                    color = Config.RISK_COLORS['low']
                elif risk < Config.RISK_MEDIUM:
                    color = Config.RISK_COLORS['medium']
                elif risk < Config.RISK_HIGH:
                    color = Config.RISK_COLORS['high']
                else:
                    color = Config.RISK_COLORS['very_high']
                
                # Check if segment is in highlighted path
                is_highlighted = False
                if highlighted_path:
                    for i in range(len(highlighted_path) - 1):
                        if (segment.start_node == highlighted_path[i] and 
                            segment.end_node == highlighted_path[i + 1]):
                            is_highlighted = True
                            break
                
                edge_trace = go.Scattermapbox(
                    lon=[start_pos[0], end_pos[0]],
                    lat=[start_pos[1], end_pos[1]],
                    mode='lines',
                    line=dict(
                        width=5 if is_highlighted else 3,
                        color='blue' if is_highlighted else color
                    ),
                    hovertext=f"Segment: {seg_id}<br>Risk: {risk:.3f}<br>"
                              f"Type: {segment.properties.get('road_type', 'N/A')}<br>"
                              f"Speed: {segment.properties.get('speed_limit', 'N/A')} km/h",
                    hoverinfo='text',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Create node trace
        node_lons = [pos[0] for pos in positions.values()]
        node_lats = [pos[1] for pos in positions.values()]
        
        node_trace = go.Scattermapbox(
            lon=node_lons,
            lat=node_lats,
            mode='markers',
            marker=dict(size=5, color='gray'),
            hoverinfo='skip',
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Calculate center
        center_lat = np.mean(node_lats)
        center_lon = np.mean(node_lons)
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            showlegend=False,
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
    
    @staticmethod
    def create_folium_map(network, highlighted_path: Optional[List[str]] = None):
        """
        Create interactive Folium map
        
        Args:
            network: NetworkManager instance
            highlighted_path: List of node IDs to highlight
        """
        positions = network.get_node_positions()
        
        if not positions:
            return None
        
        # Calculate center
        lats = [pos[1] for pos in positions.values()]
        lons = [pos[0] for pos in positions.values()]
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Add segments
        for seg_id, segment in network.segments.items():
            start_pos = positions.get(segment.start_node)
            end_pos = positions.get(segment.end_node)
            
            if start_pos and end_pos:
                risk = segment.risk_score
                
                # Determine color
                if risk < Config.RISK_LOW:
                    color = 'green'
                elif risk < Config.RISK_MEDIUM:
                    color = 'yellow'
                elif risk < Config.RISK_HIGH:
                    color = 'orange'
                else:
                    color = 'red'
                
                # Check if highlighted
                is_highlighted = False
                if highlighted_path:
                    for i in range(len(highlighted_path) - 1):
                        if (segment.start_node == highlighted_path[i] and 
                            segment.end_node == highlighted_path[i + 1]):
                            is_highlighted = True
                            color = 'blue'
                            break
                
                folium.PolyLine(
                    locations=[[start_pos[1], start_pos[0]], [end_pos[1], end_pos[0]]],
                    color=color,
                    weight=6 if is_highlighted else 4,
                    opacity=0.8,
                    popup=f"Risk: {risk:.3f}<br>Type: {segment.properties.get('road_type')}"
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
            <p style="margin: 5px;"><b>Risk Levels</b></p>
            <p style="margin: 5px;"><span style="color: green;">█</span> Low (< 0.25)</p>
            <p style="margin: 5px;"><span style="color: yellow;">█</span> Medium (0.25-0.50)</p>
            <p style="margin: 5px;"><span style="color: orange;">█</span> High (0.50-0.75)</p>
            <p style="margin: 5px;"><span style="color: red;">█</span> Very High (> 0.75)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m


class RiskHeatmap:
    """Generate risk heatmaps and statistics"""
    
    @staticmethod
    def plot_risk_distribution(network):
        """Plot histogram of risk scores"""
        risks = [seg.risk_score for seg in network.segments.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=risks,
            nbinsx=30,
            marker_color='rgb(55, 83, 109)',
            name='Risk Distribution'
        ))
        
        fig.update_layout(
            title='Road Segment Risk Distribution',
            xaxis_title='Risk Score',
            yaxis_title='Number of Segments',
            showlegend=False,
            height=400
        )
        
        # Add risk threshold lines
        fig.add_vline(x=Config.RISK_LOW, line_dash="dash", line_color="green", 
                      annotation_text="Low")
        fig.add_vline(x=Config.RISK_MEDIUM, line_dash="dash", line_color="yellow",
                      annotation_text="Medium")
        fig.add_vline(x=Config.RISK_HIGH, line_dash="dash", line_color="orange",
                      annotation_text="High")
        
        return fig
    
    @staticmethod
    def plot_risk_by_road_type(network):
        """Plot average risk by road type"""
        data = []
        for seg in network.segments.values():
            data.append({
                'road_type': seg.properties.get('road_type', 'unknown'),
                'risk': seg.risk_score
            })
        
        df = pd.DataFrame(data)
        
        fig = px.box(df, x='road_type', y='risk', 
                     title='Risk Distribution by Road Type',
                     labels={'risk': 'Risk Score', 'road_type': 'Road Type'},
                     color='road_type')
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    @staticmethod
    def plot_risk_factors(segment_features: Dict):
        """Plot radar chart of risk factors for a segment"""
        factors = {
            'Curvature': segment_features.get('curvature', 0),
            'Speed Limit': segment_features.get('speed_limit', 50) / 120,  # Normalize
            'Weather Risk': {'clear': 0, 'rainy': 0.5, 'foggy': 0.8}.get(
                segment_features.get('weather', 'clear'), 0),
            'Lighting Risk': {'daylight': 0, 'dim': 0.5, 'night': 0.8}.get(
                segment_features.get('lighting', 'daylight'), 0),
            'Accident History': min(segment_features.get('num_reported_accidents', 0) / 10, 1)
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=list(factors.values()),
            theta=list(factors.keys()),
            fill='toself',
            name='Risk Factors'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title='Risk Factor Analysis',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_importance_df: pd.DataFrame):
        """Plot feature importance from ML model"""
        if feature_importance_df.empty:
            return None
        
        fig = px.bar(
            feature_importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            labels={'importance': 'Importance Score', 'feature': 'Feature'}
        )
        
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        
        return fig
    
    @staticmethod
    def plot_route_comparison(routes: Dict):
        """Compare multiple routes visually"""
        data = []
        
        for route_type, route_info in routes.items():
            if route_info:
                data.append({
                    'Route Type': route_type.capitalize(),
                    'Distance (km)': route_info['total_distance_km'],
                    'Avg Risk': route_info['avg_risk'],
                    'Max Risk': route_info['max_risk']
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Distance (km)',
            x=df['Route Type'],
            y=df['Distance (km)'],
            yaxis='y',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            name='Avg Risk',
            x=df['Route Type'],
            y=df['Avg Risk'],
            yaxis='y2',
            marker_color='red',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title='Route Comparison',
            yaxis=dict(title='Distance (km)'),
            yaxis2=dict(title='Risk Score', overlaying='y', side='right'),
            height=400,
            hovermode='x'
        )
        
        return fig

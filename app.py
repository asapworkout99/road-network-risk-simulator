"""
Road Network Risk Simulator - Main Application
Streamlit app for accident risk prediction and route optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now import project modules - USE ABSOLUTE IMPORTS
import config
from config import Config

from models.ml_predictor import RiskPredictor, load_risk_predictor
from models.network_manager import NetworkManager, RoadSegment
from models.path_finder import PathFinder
from models.osm_integration import OSMIntegration

from utils.visualization import NetworkVisualizer, RiskHeatmap
from utils.helpers import (
    format_distance, 
    format_risk_level, 
    calculate_route_time,
    generate_risk_summary,
    get_risk_recommendations,
    validate_model_files
)

# Page configuration
st.set_page_config(
    page_title="Road Network Risk Simulator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low { color: green; font-weight: bold; }
    .risk-medium { color: orange; font-weight: bold; }
    .risk-high { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'network' not in st.session_state:
    st.session_state.network = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'selected_path' not in st.session_state:
    st.session_state.selected_path = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Road Network Risk Simulator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        st.subheader("🤖 Model Status")
        model_files = validate_model_files()
        
        if all(model_files.values()):
            st.success("✅ All model files present")
        else:
            st.warning("⚠️ Some model files missing")
            for name, exists in model_files.items():
                status = "✅" if exists else "❌"
                st.text(f"{status} {name}")
        
        # Load predictor
        if st.button("🔄 Load/Reload Model"):
            with st.spinner("Loading ML model..."):
                st.session_state.predictor = load_risk_predictor()
                if st.session_state.predictor:
                    st.success("✅ Model loaded successfully!")
                else:
                    st.warning("⚠️ Using fallback risk calculation")
        
        st.divider()
        
        # Network creation options
        st.subheader("🗺️ Network Creation")
        network_type = st.radio(
            "Choose network type:",
            ["Grid Network", "OpenStreetMap Import"],
            key="network_type"
        )
        
        if network_type == "Grid Network":
            rows = st.slider("Rows", 3, 10, 5)
            cols = st.slider("Columns", 3, 10, 5)
            spacing = st.slider("Spacing (m)", 50, 200, 100)
            
            if st.button("🏗️ Create Grid Network"):
                with st.spinner("Creating network..."):
                    network = NetworkManager()
                    network.create_grid_network(rows, cols, spacing)
                    st.session_state.network = network
                    st.success(f"✅ Created {rows}x{cols} grid network!")
        
        else:  # OSM Import
            location = st.text_input("Location", "Lagos, Nigeria")
            distance = st.slider("Radius (m)", 500, 5000, 1000, step=100)
            
            if st.button("🌍 Import from OSM"):
                with st.spinner(f"Fetching network from {location}..."):
                    osm = OSMIntegration()
                    graph = osm.fetch_network(location, distance)
                    
                    if graph:
                        network = osm.convert_osm_to_network(graph, max_segments=500)
                        st.session_state.network = network
                        st.success(f"✅ Imported network from {location}!")
                    else:
                        st.error("❌ Failed to fetch network")
    
    # Main content area
    if st.session_state.network is None:
        st.info("👈 Create or import a road network from the sidebar to get started")
        
        # Show example
        st.subheader("📊 Example Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Network Analysis**")
            st.write("- Real-time risk prediction")
            st.write("- Route optimization")
            st.write("- Safety recommendations")
        
        with col2:
            st.markdown("**Visualization**")
            st.write("- Interactive network maps")
            st.write("- Risk heatmaps")
            st.write("- Route comparison")
        
        return
    
    # Tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Network View", 
        "🎯 Risk Analysis", 
        "🛣️ Route Finding",
        "📊 Statistics"
    ])
    
    network = st.session_state.network
    
    with tab1:
        st.subheader("Network Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualize network
            if st.button("🔄 Update Risk Scores"):
                if st.session_state.predictor:
                    with st.spinner("Calculating risks..."):
                        network.update_all_risks(st.session_state.predictor)
                    st.success("✅ Risk scores updated!")
                else:
                    st.warning("⚠️ Load model first or using fallback calculation")
            
            # Show map
            fig = NetworkVisualizer.plot_network_graph(
                network, 
                st.session_state.selected_path
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            stats = network.get_statistics()
            st.metric("Nodes", stats['num_nodes'])
            st.metric("Segments", stats['num_segments'])
            st.metric("Total Length", f"{stats['total_length_km']:.2f} km")
            st.metric("Avg Risk", f"{stats['avg_risk']:.3f}")
            st.metric("High Risk Segments", stats['high_risk_segments'])
    
    with tab2:
        st.subheader("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk distribution
            fig = RiskHeatmap.plot_risk_distribution(network)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk by road type
            fig = RiskHeatmap.plot_risk_by_road_type(network)
            st.plotly_chart(fig, use_container_width=True)
        
        # High risk segments
        st.subheader("⚠️ High Risk Segments")
        path_finder = PathFinder(network)
        high_risk = path_finder.get_high_risk_segments(threshold=0.6)
        
        if high_risk:
            df = pd.DataFrame(high_risk)
            st.dataframe(df, use_container_width=True)
        else:
            st.success("✅ No high-risk segments detected!")
    
    with tab3:
        st.subheader("Route Finding")
        
        # Get available nodes
        nodes = list(network.nodes.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_node = st.selectbox("Start Node", nodes, key="start")
        
        with col2:
            end_node = st.selectbox("End Node", nodes, key="end")
        
        if st.button("🔍 Find Routes"):
            if start_node == end_node:
                st.error("❌ Start and end nodes must be different")
            else:
                with st.spinner("Finding routes..."):
                    path_finder = PathFinder(network)
                    routes = path_finder.compare_routes(start_node, end_node)
                    
                    # Display route comparison
                    route_data = []
                    for route_type, route_info in routes.items():
                        if route_info:
                            route_data.append({
                                'Route Type': route_type.capitalize(),
                                'Distance': format_distance(route_info['total_distance_m']),
                                'Est. Time': calculate_route_time(route_info['total_distance_km']),
                                'Avg Risk': f"{route_info['avg_risk']:.3f}",
                                'Max Risk': f"{route_info['max_risk']:.3f}",
                                'Segments': route_info['num_segments']
                            })
                    
                    if route_data:
                        st.success(f"✅ Found {len(route_data)} route(s)")
                        
                        # Display table
                        df = pd.DataFrame(route_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Visualization
                        fig = RiskHeatmap.plot_route_comparison(routes)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Select route to highlight
                        selected = st.radio(
                            "Select route to highlight:",
                            list(routes.keys())
                        )
                        
                        if selected and routes[selected]:
                            st.session_state.selected_path = routes[selected]['path']
                            
                            # Show recommendations
                            st.subheader("🛡️ Safety Recommendations")
                            route_info = routes[selected]
                            recommendations = get_risk_recommendations(
                                route_info['avg_risk'],
                                network.segments[list(network.segments.keys())[0]].get_features()
                            )
                            
                            for rec in recommendations:
                                st.info(rec)
                    else:
                        st.error("❌ No route found between selected nodes")
    
    with tab4:
        st.subheader("Network Statistics")
        
        summary = generate_risk_summary(network)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Segments", summary['total_segments'])
            st.metric("Average Risk", f"{summary['avg_risk']:.4f}")
            st.metric("Risk Std Dev", f"{summary['risk_std']:.4f}")
        
        with col2:
            st.metric("Min Risk", f"{summary['min_risk']:.4f}")
            st.metric("Median Risk", f"{summary['median_risk']:.4f}")
            st.metric("Max Risk", f"{summary['max_risk']:.4f}")
        
        with col3:
            st.metric("High Risk %", f"{summary['high_risk_percentage']:.1f}%")
            st.metric("Low Risk Segments", summary['risk_categories']['low'])
            st.metric("Very High Risk", summary['risk_categories']['very_high'])
        
        # Show feature importance if available
        if st.session_state.predictor:
            st.subheader("📊 Feature Importance")
            fi_df = st.session_state.predictor.get_feature_importance()
            
            if not fi_df.empty:
                fig = RiskHeatmap.plot_feature_importance(fi_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

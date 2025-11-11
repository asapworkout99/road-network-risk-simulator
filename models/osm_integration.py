"""
OpenStreetMap Integration
Fetch real road networks from OpenStreetMap using OSMnx
"""

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import streamlit as st
from models.network_manager import NetworkManager, RoadSegment

class OSMIntegration:
    """Integrate OpenStreetMap data into the road network"""
    
    def __init__(self):
        self.osm_graph = None
        self.location_name = None
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_network(location: str, distance: int = 1000, 
                     network_type: str = 'drive') -> Optional[nx.MultiDiGraph]:
        """
        Fetch road network from OpenStreetMap
        
        Args:
            location: Location name (e.g., "Lagos, Nigeria") or (lat, lon) tuple
            distance: Distance from center point in meters
            network_type: Type of network ('drive', 'walk', 'bike', 'all')
            
        Returns:
            OSMnx graph or None if failed
        """
        try:
            with st.spinner(f"🌍 Fetching road network for {location}..."):
                if isinstance(location, tuple):
                    # Fetch by coordinates
                    graph = ox.graph_from_point(
                        location, 
                        dist=distance, 
                        network_type=network_type,
                        simplify=True
                    )
                else:
                    # Fetch by place name
                    graph = ox.graph_from_place(
                        location, 
                        network_type=network_type,
                        simplify=True
                    )
                
                st.success(f"✅ Loaded {len(graph.nodes)} nodes and {len(graph.edges)} edges")
                return graph
                
        except Exception as e:
            st.error(f"❌ Error fetching OSM data: {str(e)}")
            return None
    
    def convert_osm_to_network(self, osm_graph: nx.MultiDiGraph, 
                               max_segments: int = 500) -> NetworkManager:
        """
        Convert OSMnx graph to NetworkManager
        
        Args:
            osm_graph: OSMnx graph
            max_segments: Maximum number of segments to include
            
        Returns:
            NetworkManager with road network
        """
        network = NetworkManager()
        
        # Add nodes
        for node_id, data in osm_graph.nodes(data=True):
            network.add_node(
                str(node_id),
                lat=data.get('y'),
                lon=data.get('x')
            )
        
        # Add edges (road segments)
        segment_count = 0
        for u, v, key, data in osm_graph.edges(keys=True, data=True):
            if segment_count >= max_segments:
                break
            
            # Extract road properties
            segment_id = f"OSM_{u}_{v}_{key}"
            length = data.get('length', 100)  # meters
            
            # Infer road type from OSM highway tag
            highway = data.get('highway', 'residential')
            road_type = self._map_highway_to_road_type(highway)
            
            # Estimate lanes
            lanes = int(data.get('lanes', 2)) if 'lanes' in data else self._estimate_lanes(highway)
            
            # Estimate speed limit
            maxspeed = data.get('maxspeed')
            speed_limit = self._parse_speed_limit(maxspeed, highway)
            
            # Create segment with estimated properties
            segment = RoadSegment(
                segment_id=segment_id,
                start_node=str(u),
                end_node=str(v),
                length=length,
                road_type=road_type,
                num_lanes=lanes,
                curvature=self._estimate_curvature(data),
                speed_limit=speed_limit,
                lighting='daylight',
                weather='clear',
                road_signs_present=True,
                public_road=True,
                time_of_day='afternoon',
                holiday=False,
                school_season=True,
                num_reported_accidents=0,
                osm_id=data.get('osmid', ''),
                highway_type=highway,
                name=data.get('name', 'Unnamed Road')
            )
            
            network.add_segment(segment)
            segment_count += 1
        
        return network
    
    @staticmethod
    def _map_highway_to_road_type(highway: str) -> str:
        """Map OSM highway tag to our road type"""
        highway_lower = str(highway).lower()
        
        if any(x in highway_lower for x in ['motorway', 'trunk', 'primary']):
            return 'highway'
        elif any(x in highway_lower for x in ['residential', 'living_street', 'service']):
            return 'urban'
        else:
            return 'rural'
    
    @staticmethod
    def _estimate_lanes(highway: str) -> int:
        """Estimate number of lanes from highway type"""
        highway_lower = str(highway).lower()
        
        if 'motorway' in highway_lower:
            return 3
        elif any(x in highway_lower for x in ['trunk', 'primary']):
            return 2
        else:
            return 1
    
    @staticmethod
    def _parse_speed_limit(maxspeed, highway: str) -> int:
        """Parse speed limit from OSM data"""
        if maxspeed:
            try:
                # Handle different formats: "50", "50 mph", ["50", "60"]
                if isinstance(maxspeed, list):
                    maxspeed = maxspeed[0]
                speed_str = str(maxspeed).split()[0]
                speed = int(speed_str)
                
                # Convert mph to km/h if needed
                if 'mph' in str(maxspeed).lower():
                    speed = int(speed * 1.60934)
                
                return min(speed, 120)  # Cap at 120 km/h
            except:
                pass
        
        # Default speeds by highway type
        highway_lower = str(highway).lower()
        if 'motorway' in highway_lower:
            return 100
        elif 'trunk' in highway_lower or 'primary' in highway_lower:
            return 70
        elif 'secondary' in highway_lower or 'tertiary' in highway_lower:
            return 50
        else:
            return 30
    
    @staticmethod
    def _estimate_curvature(edge_data: dict) -> float:
        """Estimate road curvature from geometry"""
        # Simple estimation: check if geometry exists
        if 'geometry' in edge_data:
            # More curved roads have more points in geometry
            geom = edge_data['geometry']
            if hasattr(geom, 'coords'):
                num_points = len(list(geom.coords))
                # Normalize: more points = more curvature
                return min(num_points / 10, 1.0)
        
        # Default to low curvature
        return np.random.uniform(0.1, 0.4)
    
    @staticmethod
    def get_city_center_coords(city_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city center"""
        city_coords = {
            'lagos': (6.5244, 3.3792),
            'abuja': (9.0765, 7.3986),
            'kano': (12.0022, 8.5919),
            'ibadan': (7.3775, 3.9470),
            'port harcourt': (4.8156, 7.0498),
            'benin city': (6.3350, 5.6037),
            'kaduna': (10.5105, 7.4165),
            'enugu': (6.4403, 7.4967),
            'jos': (9.8965, 8.8583),
            'ilorin': (8.4966, 4.5426)
        }
        
        return city_coords.get(city_name.lower())

"""
Path Finding Algorithms
Find optimal routes considering distance and risk
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from models.network_manager import NetworkManager

class PathFinder:
    """
    Find paths through road network
    Supports multiple optimization criteria
    """
    
    def __init__(self, network: NetworkManager):
        self.network = network
        self.graph = network.graph
    
    def find_shortest_path(self, start_node: str, end_node: str) -> Optional[Dict]:
        """
        Find shortest path by distance
        
        Args:
            start_node: Starting node ID
            end_node: Destination node ID
            
        Returns:
            Dictionary with path info or None if no path found
        """
        try:
            path = nx.shortest_path(
                self.graph,
                source=start_node,
                target=end_node,
                weight='length'
            )
            
            return self._analyze_path(path, 'distance')
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            print(f"Error finding shortest path: {e}")
            return None
    
    def find_safest_path(self, start_node: str, end_node: str) -> Optional[Dict]:
        """
        Find safest path (minimum risk)
        
        Args:
            start_node: Starting node ID
            end_node: Destination node ID
            
        Returns:
            Dictionary with path info or None if no path found
        """
        try:
            # Use inverse of (1 - risk) as weight to minimize risk
            # Higher risk = higher cost
            path = nx.shortest_path(
                self.graph,
                source=start_node,
                target=end_node,
                weight='risk_score'
            )
            
            return self._analyze_path(path, 'risk')
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            print(f"Error finding safest path: {e}")
            return None
    
    def find_balanced_path(self, start_node: str, end_node: str, 
                          risk_weight: float = 0.5) -> Optional[Dict]:
        """
        Find balanced path considering both distance and risk
        
        Args:
            start_node: Starting node ID
            end_node: Destination node ID
            risk_weight: Weight for risk (0-1), distance weight = 1 - risk_weight
            
        Returns:
            Dictionary with path info or None if no path found
        """
        try:
            # Create combined weight
            for u, v, key, data in self.graph.edges(keys=True, data=True):
                # Normalize length (0-1 scale)
                max_length = max(d.get('length', 100) for _, _, d in self.graph.edges(data=True))
                norm_length = data.get('length', 100) / max_length
                
                # Combine normalized distance and risk
                risk = data.get('risk_score', 0.5)
                combined_weight = (1 - risk_weight) * norm_length + risk_weight * risk
                
                self.graph[u][v][key]['combined_weight'] = combined_weight
            
            path = nx.shortest_path(
                self.graph,
                source=start_node,
                target=end_node,
                weight='combined_weight'
            )
            
            return self._analyze_path(path, 'balanced')
            
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            print(f"Error finding balanced path: {e}")
            return None
    
    def compare_routes(self, start_node: str, end_node: str) -> Dict:
        """
        Compare all three route types
        
        Returns:
            Dictionary with all three routes
        """
        return {
            'shortest': self.find_shortest_path(start_node, end_node),
            'safest': self.find_safest_path(start_node, end_node),
            'balanced': self.find_balanced_path(start_node, end_node)
        }
    
    def _analyze_path(self, path: List[str], path_type: str) -> Dict:
        """
        Analyze a path and compute statistics
        
        Args:
            path: List of node IDs in path
            path_type: Type of optimization used
            
        Returns:
            Dictionary with path statistics
        """
        total_distance = 0
        total_risk = 0
        segments = []
        risk_scores = []
        
        # Analyze each segment in path
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Get edge data
            if self.graph.has_edge(u, v):
                edge_data = self.graph[u][v]
                
                # Get first edge if multiple edges exist
                if isinstance(edge_data, dict) and len(edge_data) > 0:
                    first_key = list(edge_data.keys())[0]
                    data = edge_data[first_key]
                else:
                    data = edge_data
                
                length = data.get('length', 0)
                risk = data.get('risk_score', 0.5)
                
                total_distance += length
                total_risk += risk * length  # Weighted by length
                risk_scores.append(risk)
                
                segments.append({
                    'from': u,
                    'to': v,
                    'length': length,
                    'risk': risk,
                    'segment_id': first_key if isinstance(edge_data, dict) else None
                })
        
        # Calculate average risk (weighted by distance)
        avg_risk = total_risk / total_distance if total_distance > 0 else 0
        
        return {
            'path': path,
            'path_type': path_type,
            'num_segments': len(segments),
            'total_distance_m': total_distance,
            'total_distance_km': total_distance / 1000,
            'avg_risk': avg_risk,
            'max_risk': max(risk_scores) if risk_scores else 0,
            'min_risk': min(risk_scores) if risk_scores else 0,
            'segments': segments,
            'risk_distribution': self._categorize_risks(risk_scores)
        }
    
    @staticmethod
    def _categorize_risks(risk_scores: List[float]) -> Dict:
        """Categorize risk scores into levels"""
        if not risk_scores:
            return {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        
        return {
            'low': sum(1 for r in risk_scores if r < 0.25),
            'medium': sum(1 for r in risk_scores if 0.25 <= r < 0.50),
            'high': sum(1 for r in risk_scores if 0.50 <= r < 0.75),
            'very_high': sum(1 for r in risk_scores if r >= 0.75)
        }
    
    def get_high_risk_segments(self, threshold: float = 0.7) -> List[Dict]:
        """
        Get all segments above risk threshold
        
        Args:
            threshold: Risk threshold (0-1)
            
        Returns:
            List of high-risk segments with details
        """
        high_risk = []
        
        for seg_id, segment in self.network.segments.items():
            if segment.risk_score >= threshold:
                high_risk.append({
                    'segment_id': seg_id,
                    'risk_score': segment.risk_score,
                    'from': segment.start_node,
                    'to': segment.end_node,
                    'length': segment.length,
                    'road_type': segment.properties.get('road_type'),
                    'speed_limit': segment.properties.get('speed_limit'),
                    'num_accidents': segment.properties.get('num_reported_accidents', 0)
                })
        
        return sorted(high_risk, key=lambda x: x['risk_score'], reverse=True)
    
    def find_alternative_routes(self, start_node: str, end_node: str, 
                               k: int = 3) -> List[Dict]:
        """
        Find k alternative routes between two points
        
        Args:
            start_node: Starting node ID
            end_node: Destination node ID
            k: Number of alternative routes to find
            
        Returns:
            List of route dictionaries
        """
        try:
            # Use k-shortest paths algorithm
            paths = list(nx.shortest_simple_paths(
                self.graph,
                source=start_node,
                target=end_node,
                weight='length'
            ))
            
            # Analyze first k paths
            routes = []
            for i, path in enumerate(paths[:k]):
                route = self._analyze_path(path, f'alternative_{i+1}')
                routes.append(route)
            
            return routes
            
        except Exception as e:
            print(f"Error finding alternative routes: {e}")
            return []

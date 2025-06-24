"""
Image Segmentation Algorithm using Depth-First Search (DFS)
Handles connected component detection and labeling
"""

import numpy as np
import random
from typing import List, Tuple, Optional

class ImageSegmenter:
    """
    A class to perform image segmentation using Depth-First Search algorithm
    """
    
    def __init__(self):
        self.visited = None
        self.components = []
        self.component_count = 0
        self.rows = 0
        self.cols = 0
    
    def dfs(self, binary_img: np.ndarray, x: int, y: int, component_pixels: List[Tuple[int, int]]) -> None:
        """
        Iterative Depth-First Search to find connected components
        """
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()

            if (0 <= cx < self.rows and 0 <= cy < self.cols and 
                not self.visited[cx][cy] and binary_img[cx][cy] == 255):

                self.visited[cx][cy] = True
                component_pixels.append((cx, cy))

                # Explore 8-connected neighbors
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                            (0, 1), (1, -1), (1, 0), (1, 1)]
                
                for dx, dy in directions:
                    stack.append((cx + dx, cy + dy))

    def find_connected_components(self, binary_img: np.ndarray) -> List[List[Tuple[int, int]]]:
        """
        Find all connected components using DFS
        
        Args:
            binary_img: Binary image array
            
        Returns:
            List of components, where each component is a list of (x, y) coordinates
        """
        self.rows, self.cols = binary_img.shape
        self.visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        self.components = []
        self.component_count = 0
        
        # Scan entire image
        for i in range(self.rows):
            for j in range(self.cols):
                # If we find an unvisited white pixel, start a new component
                if binary_img[i][j] == 255 and not self.visited[i][j]:
                    component_pixels = []
                    self.dfs(binary_img, i, j, component_pixels)
                    
                    if component_pixels:  # Only add non-empty components
                        self.components.append(component_pixels)
                        self.component_count += 1
        
        return self.components
    
    def create_segmented_image(self, binary_img: np.ndarray, 
                             components: List[List[Tuple[int, int]]], 
                             color_scheme: str = 'random') -> np.ndarray:
        """
        Create a colored segmented image from components
        
        Args:
            binary_img: Original binary image
            components: List of detected components
            color_scheme: Color scheme ('random', 'rainbow', 'pastel')
            
        Returns:
            RGB image with colored components
        """
        rows, cols = binary_img.shape
        segmented_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        
        # Generate colors based on scheme
        colors = self._generate_colors(len(components), color_scheme)
        
        # Color each component
        for idx, component in enumerate(components):
            color = colors[idx]
            for x, y in component:
                segmented_img[x, y] = color
        
        return segmented_img
    
    def _generate_colors(self, num_colors: int, scheme: str = 'random') -> List[List[int]]:
        """
        Generate colors for components based on specified scheme
        
        Args:
            num_colors: Number of colors needed
            scheme: Color generation scheme
            
        Returns:
            List of RGB color values
        """
        colors = []
        
        if scheme == 'random':
            for _ in range(num_colors):
                color = [random.randint(50, 255) for _ in range(3)]
                colors.append(color)
        
        elif scheme == 'rainbow':
            import colorsys
            for i in range(num_colors):
                hue = i / max(num_colors, 1)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = [int(c * 255) for c in rgb]
                colors.append(color)
        
        elif scheme == 'pastel':
            pastel_colors = [
                [255, 179, 186], [255, 223, 186], [255, 255, 186],
                [186, 255, 201], [186, 225, 255], [186, 186, 255],
                [255, 186, 255], [255, 186, 223], [201, 255, 255],
                [255, 201, 201], [201, 201, 255], [201, 255, 201]
            ]
            colors = pastel_colors * (num_colors // len(pastel_colors) + 1)
            colors = colors[:num_colors]
        
        return colors
    
    def get_component_stats(self, components: List[List[Tuple[int, int]]]) -> dict:
        """
        Calculate statistics for detected components
        
        Args:
            components: List of detected components
            
        Returns:
            Dictionary with component statistics
        """
        if not components:
            return {
                'total_components': 0,
                'total_pixels': 0,
                'avg_component_size': 0,
                'largest_component': 0,
                'smallest_component': 0
            }
        
        component_sizes = [len(comp) for comp in components]
        total_pixels = sum(component_sizes)
        
        return {
            'total_components': len(components),
            'total_pixels': total_pixels,
            'avg_component_size': total_pixels / len(components),
            'largest_component': max(component_sizes),
            'smallest_component': min(component_sizes),
            'component_sizes': component_sizes
        }
    
    def filter_components_by_size(self, components: List[List[Tuple[int, int]]], 
                                 min_size: int = 1, 
                                 max_size: Optional[int] = None) -> List[List[Tuple[int, int]]]:
        """
        Filter components based on size criteria
        
        Args:
            components: List of components to filter
            min_size: Minimum component size
            max_size: Maximum component size (None for no limit)
            
        Returns:
            Filtered list of components
        """
        filtered = []
        for comp in components:
            size = len(comp)
            if size >= min_size:
                if max_size is None or size <= max_size:
                    filtered.append(comp)
        
        return filtered
    
    def get_component_bounding_boxes(self, components: List[List[Tuple[int, int]]]) -> List[dict]:
        """
        Calculate bounding boxes for each component
        
        Args:
            components: List of detected components
            
        Returns:
            List of bounding box dictionaries
        """
        bounding_boxes = []
        
        for i, component in enumerate(components):
            if not component:
                continue
                
            x_coords = [x for x, y in component]
            y_coords = [y for x, y in component]
            
            bbox = {
                'component_id': i,
                'min_x': min(x_coords),
                'max_x': max(x_coords),
                'min_y': min(y_coords),
                'max_y': max(y_coords),
                'width': max(y_coords) - min(y_coords) + 1,
                'height': max(x_coords) - min(x_coords) + 1,
                'center_x': (min(x_coords) + max(x_coords)) / 2,
                'center_y': (min(y_coords) + max(y_coords)) / 2
            }
            bounding_boxes.append(bbox)
        
        return bounding_boxes
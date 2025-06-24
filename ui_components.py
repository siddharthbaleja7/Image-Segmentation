"""
Streamlit UI components and layout functions
Handles all user interface elements and interactions
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

class UIComponents:
    """
    Class containing all UI component functions
    """
    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Image Segmentation with DFS",
            page_icon="🖼️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def render_header():
        """Render the main page header"""
        st.title("🖼️ Interactive Image Segmentation using DFS")
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 2rem; color: white;'>
            <h3>Detect Connected Components using Depth-First Search Algorithm</h3>
            <p>Upload binary or grayscale images to identify and segment connected regions</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    @staticmethod
    def render_sidebar():
        """Render the information sidebar"""
        with st.sidebar:
            st.header("📖 About This Tool")
            st.markdown("""
            This application uses **Depth-First Search (DFS)** to identify 
            connected components in binary images. Each connected region 
            receives a unique color for easy visualization.
            """)
            
            st.header("🎯 Applications")
            applications = {
                "🏥 Medical Imaging": "Detect cells, tumors, organs",
                "📝 OCR Processing": "Separate text characters",
                "🤖 Robotics": "Object recognition",
                "📹 Surveillance": "Motion detection",
                "🌾 Agriculture": "Crop analysis"
            }
            
            for app, desc in applications.items():
                st.markdown(f"**{app}**: {desc}")
            
            st.header("📋 How to Use")
            steps = [
                "Upload binary/grayscale image",
                "Adjust threshold if needed",
                "Apply morphological operations",
                "View segmentation results",
                "Download processed images",
                "Analyze component statistics"
            ]
            
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")
            
            st.header("⚙️ Algorithm Details")
            st.info("""
            **DFS Traversal**: Explores 8-connected neighbors recursively
            
            **Time Complexity**: O(n) where n = pixels
            
            **Space Complexity**: O(n) for visited matrix
            """)
    
    @staticmethod
    def render_file_uploader():
        """Render file upload section"""
        st.header("📤 Upload Image")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload binary, grayscale, or color images for segmentation"
            )
        
        with col2:
            if uploaded_file:
                file_size = len(uploaded_file.getvalue()) / 1024  # KB
                st.metric("File Size", f"{file_size:.1f} KB")
        
        return uploaded_file
    
    @staticmethod
    def render_image_info(image: np.ndarray, title: str = "Image Info"):
        """Render image information panel"""
        st.subheader(title)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Width", f"{image.shape[1]} px")
        with col2:
            st.metric("Height", f"{image.shape[0]} px")
        with col3:
            channels = len(image.shape)
            st.metric("Channels", channels)
        with col4:
            size_mb = image.nbytes / (1024 * 1024)
            st.metric("Size", f"{size_mb:.2f} MB")
    
    @staticmethod
    def render_preprocessing_controls():
        """Render preprocessing control panel"""
        st.header("🔧 Preprocessing Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Binary Threshold")
            threshold = st.slider(
                "Threshold Value",
                min_value=0,
                max_value=255,
                value=127,
                help="Pixels above this value become white (255)"
            )
            
            invert = st.checkbox(
                "Invert Binary",
                help="Swap black and white regions"
            )
        
        with col2:
            st.subheader("Morphological Operations")
            morph_op = st.selectbox(
                "Operation Type",
                options=['none', 'opening', 'closing', 'erosion', 'dilation'],
                help="Clean up binary image before segmentation"
            )
            
            if morph_op != 'none':
                kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2)
                iterations = st.slider("Iterations", 1, 5, 1)
            else:
                kernel_size, iterations = 3, 1
        
        return threshold, invert, morph_op, kernel_size, iterations
    
    @staticmethod
    def render_segmentation_controls():
        """Render segmentation control panel"""
        st.header("🎨 Segmentation Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            color_scheme = st.selectbox(
                "Color Scheme",
                options=['random', 'rainbow', 'pastel'],
                help="Choose how components are colored"
            )
        
        with col2:
            min_component_size = st.number_input(
                "Minimum Component Size",
                min_value=1,
                max_value=10000,
                value=10,
                help="Filter out components smaller than this"
            )
        
        return color_scheme, min_component_size
    
    @staticmethod
    def render_results_display(original_img, binary_img, segmented_img, 
                             components, stats):
        """Render segmentation results"""
        st.header("🎨 Segmentation Results")
        
        # Image display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_img, use_column_width=True)
        
        with col2:
            st.subheader("Binary Image")
            st.image(binary_img, use_column_width=True)
        
        with col3:
            st.subheader("Segmented Image")
            st.image(segmented_img, use_column_width=True)
        
        # Statistics display
        st.subheader("📊 Segmentation Statistics")
        UIComponents.render_statistics_panel(stats)
        
        # Component details
        if components:
            UIComponents.render_component_table(components, stats)
    
    @staticmethod
    def render_statistics_panel(stats: dict):
        """Render statistics metrics panel"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Components", stats['total_components'])
        with col2:
            st.metric("Total Pixels", f"{stats['total_pixels']:,}")
        with col3:
            st.metric("Avg Size", f"{stats['avg_component_size']:.0f}")
        with col4:
            st.metric("Largest", f"{stats['largest_component']:,}")
        with col5:
            st.metric("Smallest", f"{stats['smallest_component']:,}")
    
    @staticmethod
    def render_component_table(components: List, stats: dict):
        """Render detailed component information table"""
        st.subheader("📋 Component Details")
        
        # Create DataFrame
        data = []
        for i, comp in enumerate(components):
            size = len(comp)
            percentage = (size / stats['total_pixels']) * 100
            data.append({
                "Component ID": i + 1,
                "Size (pixels)": f"{size:,}",
                "Percentage": f"{percentage:.2f}%",
                "Relative Size": "●" * min(int(percentage), 20)
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def render_component_visualization(stats: dict):
        """Render component size visualization"""
        if stats['total_components'] == 0:
            return
        
        st.subheader("📈 Component Size Distribution")
        
        # Create histogram
        sizes = stats['component_sizes']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                x=sizes,
                nbins=min(20, len(sizes)),
                title="Component Size Distribution",
                labels={'x': 'Component Size (pixels)', 'y': 'Count'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                y=sizes,
                title="Component Size Statistics",
                labels={'y': 'Component Size (pixels)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
    
    @staticmethod
    def render_download_section(segmented_img, binary_img, components, stats):
        """Render download options"""
        st.header("💾 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Segmented Image", key="seg_download"):
                from utils import FileManager
                download_link = FileManager.get_image_download_link(
                    segmented_img, "segmented_image.png"
                )
                st.markdown(download_link, unsafe_allow_html=True)
        
        with col2:
            if st.button("📥 Download Binary Image", key="bin_download"):
                from utils import FileManager
                binary_rgb = np.stack([binary_img] * 3, axis=-1)
                download_link = FileManager.get_image_download_link(
                    binary_rgb, "binary_image.png"
                )
                st.markdown(download_link, unsafe_allow_html=True)
        
        with col3:
            if st.button("📊 Download CSV Data", key="csv_download"):
                from utils import FileManager
                csv_link = FileManager.save_component_data_as_csv(
                    components, stats, "component_data.csv"
                )
                st.markdown(csv_link, unsafe_allow_html=True)
    
    @staticmethod
    def render_advanced_analysis(components, binary_img):
        """Render advanced analysis section"""
        if not components:
            return
        
        st.header("🔬 Advanced Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Bounding Boxes", "Shape Analysis", "Spatial Distribution"])
        
        with tab1:
            UIComponents.render_bounding_boxes_analysis(components)
        
        with tab2:
            UIComponents.render_shape_analysis(components)
        
        with tab3:
            UIComponents.render_spatial_analysis(components, binary_img.shape)
    
    @staticmethod
    def render_bounding_boxes_analysis(components):
        """Render bounding boxes analysis"""
        from segmentation import ImageSegmenter
        
        segmenter = ImageSegmenter()
        bboxes = segmenter.get_component_bounding_boxes(components)
        
        if not bboxes:
            st.info("No components found for bounding box analysis")
            return
        
        # Create DataFrame
        bbox_data = []
        for bbox in bboxes:
            bbox_data.append({
                "Component": bbox['component_id'] + 1,
                "Width": bbox['width'],
                "Height": bbox['height'],
                "Area": bbox['width'] * bbox['height'],
                "Center X": f"{bbox['center_x']:.1f}",
                "Center Y": f"{bbox['center_y']:.1f}",
                "Aspect Ratio": f"{bbox['width']/bbox['height']:.2f}"
            })
        
        df = pd.DataFrame(bbox_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def render_shape_analysis(components):
        """Render shape analysis"""
        if not components:
            return
        
        st.subheader("Shape Characteristics")
        
        # Calculate shape metrics
        shape_data = []
        for i, comp in enumerate(components):
            if len(comp) < 3:  # Need at least 3 points
                continue
            
            # Calculate perimeter (approximate)
            perimeter = UIComponents._calculate_perimeter(comp)
            area = len(comp)
            
            # Compactness (circular = 1, elongated < 1)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            shape_data.append({
                "Component": i + 1,
                "Area": area,
                "Perimeter": f"{perimeter:.1f}",
                "Compactness": f"{compactness:.3f}",
                "Shape": "Circular" if compactness > 0.7 else "Elongated"
            })
        
        if shape_data:
            df = pd.DataFrame(shape_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def render_spatial_analysis(components, image_shape):
        """Render spatial distribution analysis"""
        if not components:
            return
        
        st.subheader("Spatial Distribution")
        
        # Calculate centroids
        centroids = []
        sizes = []
        
        for comp in components:
            if comp:
                x_coords = [x for x, y in comp]
                y_coords = [y for x, y in comp]
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                centroids.append((centroid_x, centroid_y))
                sizes.append(len(comp))
        
        if centroids:
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[c[1] for c in centroids],
                y=[image_shape[0] - c[0] for c in centroids],  # Flip Y for display
                mode='markers',
                marker=dict(
                    size=[np.sqrt(s) for s in sizes],
                    color=sizes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Component Size")
                ),
                text=[f"Component {i+1}<br>Size: {s}" for i, s in enumerate(sizes)],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Component Spatial Distribution",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                width=600,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _calculate_perimeter(component):
        """Calculate approximate perimeter of a component"""
        if len(component) < 2:
            return 0
        
        # Find boundary pixels (simplified)
        boundary_pixels = set()
        
        for x, y in component:
            # Check if pixel is on boundary (has non-component neighbor)
            neighbors = [
                (x-1, y-1), (x-1, y), (x-1, y+1),
                (x, y-1),              (x, y+1),
                (x+1, y-1), (x+1, y), (x+1, y+1)
            ]
            
            for nx, ny in neighbors:
                if (nx, ny) not in component:
                    boundary_pixels.add((x, y))
                    break
        
        return len(boundary_pixels)
    
    @staticmethod
    def render_error_message(error_msg: str, error_type: str = "error"):
        """Render error or warning messages"""
        if error_type == "error":
            st.error(f"❌ {error_msg}")
        elif error_type == "warning":
            st.warning(f"⚠️ {error_msg}")
        elif error_type == "info":
            st.info(f"ℹ️ {error_msg}")
    
    @staticmethod
    def render_processing_status(status: str, progress: float = None):
        """Render processing status with optional progress bar"""
        if progress is not None:
            st.progress(progress)
        
        with st.spinner(status):
            pass
    
    @staticmethod
    def render_footer():
        """Render page footer"""
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; padding: 2rem; background: #f0f2f6; border-radius: 10px;'>
                <h4>🖼️ Image Segmentation with DFS</h4>
                <p>Built with ❤️ using Streamlit | Connected Components Detection</p>
                <p><strong>Algorithm:</strong> Depth-First Search (DFS) | <strong>Time Complexity:</strong> O(n)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
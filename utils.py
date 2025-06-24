"""
Image processing utilities and helper functions
Handles image preprocessing, file operations, and conversions
"""

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from io import BytesIO
import base64
from typing import Tuple, Optional, Union

class ImageProcessor:
    """
    Utility class for image processing operations
    """
    
    @staticmethod
    def load_image_from_upload(uploaded_file) -> np.ndarray:
        """
        Load image from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Image as numpy array
        """
        try:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            return img_array
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_image(image: np.ndarray, 
                        threshold: int = 127, 
                        invert: bool = False) -> np.ndarray:
        """
        Convert image to binary format
        
        Args:
            image: Input image array
            threshold: Binary threshold value (0-255)
            invert: Whether to invert the binary image
            
        Returns:
            Binary image array
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold
        if invert:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        return binary
    
    @staticmethod
    def apply_morphological_operations(binary_img: np.ndarray, 
                                     operation: str = 'none',
                                     kernel_size: int = 3,
                                     iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to clean up binary image
        
        Args:
            binary_img: Binary image array
            operation: Type of operation ('opening', 'closing', 'erosion', 'dilation', 'none')
            kernel_size: Size of morphological kernel
            iterations: Number of iterations
            
        Returns:
            Processed binary image
        """
        if operation == 'none':
            return binary_img
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'erosion':
            return cv2.erode(binary_img, kernel, iterations=iterations)
        elif operation == 'dilation':
            return cv2.dilate(binary_img, kernel, iterations=iterations)
        elif operation == 'opening':
            return cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'closing':
            return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            return binary_img
    
    @staticmethod
    def resize_image(image: np.ndarray, 
                    max_width: int = 800, 
                    max_height: int = 600) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: Input image array
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image array
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return resized
        
        return image
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """
        Get detailed information about an image
        
        Args:
            image: Image array
            
        Returns:
            Dictionary with image information
        """
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'size_mb': image.nbytes / (1024 * 1024),
            'channels': len(image.shape),
            'total_pixels': image.shape[0] * image.shape[1]
        }
        
        if len(image.shape) == 2:  # Grayscale
            info['min_value'] = int(np.min(image))
            info['max_value'] = int(np.max(image))
            info['mean_value'] = float(np.mean(image))
            info['unique_values'] = len(np.unique(image))
        
        return info

class FileManager:
    """
    Utility class for file operations and downloads
    """
    
    @staticmethod
    def get_image_download_link(img_array: np.ndarray, 
                               filename: str,
                               format: str = "PNG") -> str:
        """
        Generate download link for processed image
        
        Args:
            img_array: Image array to download
            filename: Name for downloaded file
            format: Image format (PNG, JPEG)
            
        Returns:
            HTML download link
        """
        try:
            img = Image.fromarray(img_array)
            buffered = BytesIO()
            img.save(buffered, format=format)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            href = f'''
            <a href="data:file/{format.lower()};base64,{img_str}" 
               download="{filename}" 
               style="text-decoration: none; 
                      background-color: #ff4b4b; 
                      color: white; 
                      padding: 0.5rem 1rem; 
                      border-radius: 0.25rem; 
                      font-weight: bold;">
                📥 Download {filename}
            </a>
            '''
            return href
        except Exception as e:
            st.error(f"Error creating download link: {str(e)}")
            return ""
    
    @staticmethod
    def save_component_data_as_csv(components: list, 
                                  component_stats: dict, 
                                  filename: str = "component_data.csv") -> str:
        """
        Create CSV download link for component data
        
        Args:
            components: List of detected components
            component_stats: Component statistics dictionary
            filename: CSV filename
            
        Returns:
            HTML download link for CSV
        """
        try:
            import pandas as pd
            
            # Create DataFrame with component information
            data = []
            for i, comp in enumerate(components):
                data.append({
                    'Component_ID': i + 1,
                    'Size_Pixels': len(comp),
                    'Percentage': f"{(len(comp) / component_stats['total_pixels'] * 100):.2f}%"
                })
            
            df = pd.DataFrame(data)
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_str = base64.b64encode(csv_buffer.getvalue()).decode()
            
            href = f'''
            <a href="data:text/csv;base64,{csv_str}" 
               download="{filename}"
               style="text-decoration: none; 
                      background-color: #00cc00; 
                      color: white; 
                      padding: 0.5rem 1rem; 
                      border-radius: 0.25rem; 
                      font-weight: bold;">
                📊 Download CSV Data
            </a>
            '''
            return href
        except ImportError:
            st.warning("Pandas not available. CSV export disabled.")
            return ""
        except Exception as e:
            st.error(f"Error creating CSV: {str(e)}")
            return ""

class ValidationUtils:
    """
    Utility class for input validation and error handling
    """
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate uploaded image
        
        Args:
            image: Image array to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"
        
        if len(image.shape) not in [2, 3]:
            return False, "Image must be 2D (grayscale) or 3D (color)"
        
        if image.shape[0] == 0 or image.shape[1] == 0:
            return False, "Image has zero dimensions"
        
        # Check for reasonable size limits
        max_pixels = 10_000_000  # 10 megapixels
        total_pixels = image.shape[0] * image.shape[1]
        if total_pixels > max_pixels:
            return False, f"Image too large ({total_pixels:,} pixels). Maximum: {max_pixels:,}"
        
        return True, "Image is valid"
    
    @staticmethod
    def validate_threshold(threshold: int) -> Tuple[bool, str]:
        """
        Validate threshold value
        
        Args:
            threshold: Threshold value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(threshold, int):
            return False, "Threshold must be an integer"
        
        if threshold < 0 or threshold > 255:
            return False, "Threshold must be between 0 and 255"
        
        return True, "Threshold is valid"
    
    @staticmethod
    def check_binary_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        Check if image is properly binary
        
        Args:
            image: Image array to check
            
        Returns:
            Tuple of (is_binary, message)
        """
        unique_values = np.unique(image)
        
        if len(unique_values) == 1:
            return False, f"Image has only one value: {unique_values[0]}"
        
        if len(unique_values) == 2 and set(unique_values) == {0, 255}:
            return True, "Image is properly binary (0 and 255)"
        
        if len(unique_values) <= 10:
            return True, f"Image has {len(unique_values)} unique values: {unique_values}"
        
        return False, f"Image has {len(unique_values)} unique values - may need thresholding"
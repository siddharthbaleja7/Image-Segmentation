"""
Configuration settings for the Image Segmentation application
"""

# Image processing settings
MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 900
MAX_FILE_SIZE_MB = 10
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']

# Algorithm settings
DEFAULT_THRESHOLD = 127
DEFAULT_MIN_COMPONENT_SIZE = 10
MAX_COMPONENTS_DISPLAY = 1000

# UI settings
PAGE_TITLE = "Image Segmentation with DFS"
PAGE_ICON = "🖼️"
LAYOUT = "wide"

# Color schemes
COLOR_SCHEMES = {
    'random': 'Random bright colors',
    'rainbow': 'Rainbow spectrum',
    'pastel': 'Soft pastel colors'
}

# Morphological operations
MORPH_OPERATIONS = {
    'none': 'No operation',
    'opening': 'Remove noise (erosion + dilation)',
    'closing': 'Fill gaps (dilation + erosion)', 
    'erosion': 'Shrink objects',
    'dilation': 'Expand objects'
}
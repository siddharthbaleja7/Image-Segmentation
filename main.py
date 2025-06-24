import streamlit as st
import numpy as np
from segmentation import ImageSegmenter
from utils import ImageProcessor, FileManager, ValidationUtils
from ui_components import UIComponents

def main():
    """Main application function"""
    
    UIComponents.setup_page_config()
    UIComponents.render_header()
    UIComponents.render_sidebar()
    
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    
    uploaded_file = UIComponents.render_file_uploader()
    
    if uploaded_file is not None:
        try:
            original_img = ImageProcessor.load_image_from_upload(uploaded_file)
            if original_img is None:
                UIComponents.render_error_message("Failed to load image", "error")
                return

            is_valid, error_msg = ValidationUtils.validate_image(original_img)
            if not is_valid:
                UIComponents.render_error_message(error_msg, "error")
                return

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📷 Original Image")
                st.image(original_img, caption="Uploaded Image", use_column_width=True)
            with col2:
                UIComponents.render_image_info(original_img, "📊 Image Information")

            st.markdown("---")
            color_scheme, min_component_size = UIComponents.render_segmentation_controls()

            if st.button("🚀 Start Segmentation", type="primary"):
                process_image(original_img, color_scheme, min_component_size)

            if st.session_state.processed_results is not None:
                display_results(st.session_state.processed_results)

        except Exception as e:
            UIComponents.render_error_message(f"Unexpected error: {str(e)}", "error")
            st.exception(e)
    else:
        render_welcome_section()

    UIComponents.render_footer()

def process_image(original_img, color_scheme, min_component_size):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Preprocessing image...")
        progress_bar.progress(20)

        processed_img = ImageProcessor.resize_image(original_img, max_width=1200, max_height=900)
        binary_img = ImageProcessor.preprocess_image(processed_img)

        progress_bar.progress(40)

        status_text.text("🔍 Validating binary image...")
        is_binary, binary_msg = ValidationUtils.check_binary_image(binary_img)
        if not is_binary:
            st.warning(f"Image validation: {binary_msg}")

        progress_bar.progress(50)

        status_text.text("🎨 Finding connected components...")
        segmenter = ImageSegmenter()
        components = segmenter.find_connected_components(binary_img)

        progress_bar.progress(70)

        status_text.text("🔧 Filtering components...")
        if min_component_size > 1:
            components = segmenter.filter_components_by_size(components, min_component_size)

        progress_bar.progress(80)

        status_text.text("🌈 Creating segmented image...")
        if len(components) == 0:
            st.warning("No components found after filtering.")
            return
        segmented_img = segmenter.create_segmented_image(binary_img, components, color_scheme)

        progress_bar.progress(90)

        status_text.text("📊 Calculating statistics...")
        stats = segmenter.get_component_stats(components)

        progress_bar.progress(100)

        st.session_state.processed_results = {
            'original_img': processed_img,
            'binary_img': binary_img,
            'segmented_img': segmented_img,
            'components': components,
            'stats': stats,
            'parameters': {
                'color_scheme': color_scheme,
                'min_component_size': min_component_size
            }
        }

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Segmentation complete! Found {len(components)} components.")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)

def display_results(results):
    st.markdown("---")

    UIComponents.render_results_display(
        results['original_img'],
        results['binary_img'], 
        results['segmented_img'],
        results['components'],
        results['stats']
    )

    if results['stats']['total_components'] > 0:
        UIComponents.render_component_visualization(results['stats'])
        UIComponents.render_advanced_analysis(
            results['components'], 
            results['binary_img']
        )

    UIComponents.render_download_section(
        results['segmented_img'],
        results['binary_img'],
        results['components'],
        results['stats']
    )

    render_processing_summary(results['parameters'])

def render_processing_summary(parameters):
    st.header("⚙️ Processing Parameters")
    st.subheader("Segmentation")
    st.write(f"**Color Scheme:** {parameters['color_scheme']}")
    st.write(f"**Min Component Size:** {parameters['min_component_size']}")

def render_welcome_section():
    st.markdown("---")
    st.header("🎯 Welcome to Image Segmentation with DFS!")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 🚀 Getting Started
        1. **Upload an image**
        2. **Choose segmentation color settings**
        3. **Click "Start Segmentation"**
        4. **Analyze results & download**
        
        ### 📋 Supported Image Types
        - Binary, Grayscale, Color (auto-converted)
        - Formats: PNG, JPG, JPEG, BMP, TIFF
        """)
    with col2:
        st.markdown("""
        ### 📊 Sample Output
        - 🎨 Colored segments
        - 📈 Component stats
        - 💾 Download images
        - 🧠 Bounding box & shape analysis
        """)

    st.markdown("---")
    st.header("💡 Example Use Cases")

    examples = [
        {
            "title": "🏥 Medical Imaging",
            "description": "Segment cells in microscopy images",
            "use_case": "Cell counting, tumor detection"
        },
        {
            "title": "📝 Document Processing", 
            "description": "Separate characters in scanned text",
            "use_case": "OCR preprocessing, text analysis"
        },
        {
            "title": "🤖 Computer Vision",
            "description": "Object detection and recognition",
            "use_case": "Robot navigation, quality control"
        },
        {
            "title": "🌾 Agriculture",
            "description": "Analyze crop health from satellite images", 
            "use_case": "Precision farming, yield prediction"
        }
    ]

    cols = st.columns(len(examples))
    for i, example in enumerate(examples):
        with cols[i]:
            st.markdown(f"""
            <div style='padding: 1rem; border: 1px solid #ddd; border-radius: 5px; height: 150px;'>
                <h4>{example['title']}</h4>
                <p><strong>Application:</strong> {example['description']}</p>
                <p><strong>Use Case:</strong> {example['use_case']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

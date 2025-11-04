import streamlit as st
from PIL import Image, ImageChops
import io
import os
import numpy as np

# --- Utility Functions (from background_remover) ---

def color_distance(c1, c2):
    """Calculates Euclidean distance between two RGB colors."""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def hex_to_rgb(hex_color):
    """Converts a hex color string to an RGB tuple."""
    try:
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        # Fallback to white for invalid hex
        return (255, 255, 255)

def find_bounding_box(img_rgba):
    """Finds the tight bounding box of the non-transparent (object) area."""
    if img_rgba.mode != 'RGBA':
        img_rgba = img_rgba.convert('RGBA')
    
    alpha_channel = img_rgba.split()[-1]
    bbox = alpha_channel.getbbox()
    
    if bbox is None:
        return (0, 0, img_rgba.width, img_rgba.height)
    return bbox

# --- Core Functions (Combined and Adapted for Streamlit) ---

def process_and_place_on_canvas(image_file, final_bg_color, tolerance, new_width, new_height, occupancy_percent, skip_bg_removal):
    """
    Background Remover and Canvas Processor logic (from background_remover.py)
    """
    img_rgb = Image.open(image_file).convert("RGB")
    img_rgba = img_rgb.convert("RGBA")
    
    # 1. Conditional Background Processing (Make Transparent)
    if not skip_bg_removal:
        pixels_rgba = img_rgba.load()
        try:
            # Assume top-left corner is the background color
            original_bg_color = img_rgb.getpixel((0, 0)) 
        except IndexError:
            st.error("Image size is too small for background detection.")
            return None

        for x in range(img_rgba.width):
            for y in range(img_rgba.height):
                current_pixel_rgb = img_rgb.getpixel((x, y))
                if color_distance(current_pixel_rgb, original_bg_color) < tolerance:
                    pixels_rgba[x, y] = (0, 0, 0, 0) 

    # 2. Calculate Object Surface Area / Bounding Box (BBOX)
    left, upper, right, lower = find_bounding_box(img_rgba)
    object_cropped = img_rgba.crop((left, upper, right, lower))
    object_width = object_cropped.width
    object_height = object_cropped.height

    if object_width == 0 or object_height == 0:
        st.warning("Object area is zero after background removal. Outputting original image on canvas.")
        object_cropped = img_rgba

    # 3. Calculate Target Size and Resize (Zoom)
    target_max_dim = min(new_width, new_height)
    target_object_max_size = int(target_max_dim * (occupancy_percent / 100.0))

    aspect_ratio = object_cropped.width / object_cropped.height

    if aspect_ratio > 1: # Wider than tall
        new_object_width = target_object_max_size
        new_object_height = int(new_object_width / aspect_ratio)
    else: # Taller than wide
        new_object_height = target_object_max_size
        new_object_width = int(new_object_height * aspect_ratio)
    
    new_object_size = (new_object_width, new_object_height)
    
    # Use LANCZOS for high-quality resizing
    resized_object = object_cropped.resize(new_object_size, Image.LANCZOS)

    # 4. Create new canvas and paste
    final_bg_rgb = hex_to_rgb(final_bg_color)
    new_canvas = Image.new("RGB", (new_width, new_height), final_bg_rgb)
    
    x_offset = (new_width - new_object_width) // 2
    y_offset = (new_height - new_object_height) // 2
    
    # Paste using the object's alpha mask
    new_canvas.paste(resized_object, (x_offset, y_offset), resized_object)

    return new_canvas

def generate_mockup(dummy_file, design_file, bg_hex, tshirt_hex, blend_factor, design_scale, offset_x, offset_y):
    """
    Design Mockup logic (from app with background change.py)
    """
    # 1. Background Canvas Setup
    dummy = Image.open(dummy_file).convert("RGBA")
    bg_r, bg_g, bg_b = hex_to_rgb(bg_hex)
    final_img = Image.new('RGB', dummy.size, (bg_r, bg_g, bg_b))
    
    # 2. T-shirt Color Change and Blending
    tshirt_r, tshirt_g, tshirt_b = hex_to_rgb(tshirt_hex)
    tshirt_color_img = Image.new('RGB', dummy.size, (tshirt_r, tshirt_g, tshirt_b))
    
    # Blend the color with dummy to retain shadows/folds
    blended_tshirt = Image.blend(dummy.convert('RGB'), tshirt_color_img, blend_factor / 100.0)
    
    # Paste the blended t-shirt onto the background using the dummy's alpha mask
    final_img.paste(blended_tshirt, (0, 0), dummy)
    
    # 3. Place Design on T-shirt
    design = Image.open(design_file).convert("RGBA")
    
    # Calculate size and position
    scale = design_scale / 100
    # Original logic: design_width = int(final_img.width * scale * 0.5)
    design_width = int(final_img.width * scale * 0.5) 
    aspect_ratio = design.height / design.width
    design_height = int(design_width * aspect_ratio)
    design = design.resize((design_width, design_height), Image.LANCZOS)
    
    max_offset_x = final_img.width - design.width
    max_offset_y = final_img.height - design_height
    pos_x = int((offset_x / 100) * max_offset_x)
    pos_y = int((offset_y / 100) * max_offset_y)

    # Create a canvas for the design
    design_canvas = Image.new('RGBA', final_img.size, (0, 0, 0, 0))
    design_canvas.paste(design, (pos_x, pos_y), design)

    # Blend the design with the t-shirt to incorporate shadows/texture (using original dummy alpha as mask)
    # The original logic was complex (composite with mask). A simpler overlay/paste is used here,
    # as PIL's composite function is not ideal for overlay blending.
    # The original alpha-masking logic from Tkinter is replicated for design placement.
    final_img.paste(design_canvas, (0, 0), design_canvas) # Paste design with its alpha channel

    return final_img

def convert_image_to_bytes(img):
    """Converts a PIL Image object to bytes for Streamlit download."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Image Processing Hub")

st.title("👕 Image Processing Hub (Streamlit)")
st.write("---")

# Use a sidebar for navigation (like a tab)
tool_selection = st.sidebar.radio(
    "Select Tool",
    ("1. Background Remover / Canvas", "2. Design Mockup Tool")
)

# --- Tool 1: Background Remover / Canvas Processor ---
if tool_selection == "1. Background Remover / Canvas":
    st.header("🖼️ Background Remover / Canvas Processor")
    st.write("Upload an image to remove its background (based on top-left color) and place it on a new canvas.")

    uploaded_file = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    with st.sidebar.expander("Canvas Settings"):
        canvas_width = st.number_input("Canvas Width (px)", value=1200, min_value=100)
        canvas_height = st.number_input("Canvas Height (px)", value=1200, min_value=100)
        occupancy = st.slider("Object Occupancy (%)", 50, 100, 95)
        bg_color = st.color_picker("Final Background Color", "#FFFFFF")
        
    with st.sidebar.expander("BG Removal Settings"):
        tolerance = st.slider("BG Removal Tolerance (0-200)", 0, 200, 30)
        skip_bg_removal = st.checkbox("Skip Background Removal", value=False)
    
    if uploaded_file is not None:
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                output_image = process_and_place_on_canvas(
                    image_file=uploaded_file,
                    final_bg_color=bg_color,
                    tolerance=tolerance,
                    new_width=canvas_width,
                    new_height=canvas_height,
                    occupancy_percent=occupancy,
                    skip_bg_removal=skip_bg_removal
                )
            
            if output_image:
                st.success("Processing Complete!")
                
                col1, col2 = st.columns(2)
                col1.subheader("Original Image")
                col1.image(Image.open(uploaded_file), use_column_width=True)
                
                col2.subheader("Processed Canvas Output")
                col2.image(output_image, use_column_width=True)
                
                st.download_button(
                    label="Download Processed PNG",
                    data=convert_image_to_bytes(output_image),
                    file_name="canvas_output.png",
                    mime="image/png"
                )

# --- Tool 2: Design Mockup Tool ---
elif tool_selection == "2. Design Mockup Tool":
    st.header("👕 Design Mockup Tool (T-shirt/Product)")
    st.write("Upload a product image (like a t-shirt, PNG) and a design/logo to create a colored mockup.")
    
    col_files = st.columns(2)
    dummy_file = col_files[0].file_uploader("1. Upload Product/Dummy Image (PNG)", type=["png"], key="dummy_upload")
    design_file = col_files[1].file_uploader("2. Upload Design/Logo Image (PNG)", type=["png"], key="design_upload")

    # Parameters in a sidebar for a clean look
    with st.sidebar.expander("Color & Blending Settings"):
        bg_color = st.color_picker("Background Color", "#FFFFFF", key="mockup_bg")
        tshirt_color = st.color_picker("T-shirt Color", "#0000FF", key="mockup_tshirt")
        blending_factor = st.slider("T-shirt Blending Factor (%)", 10, 100, 70)
        
    with st.sidebar.expander("Design Placement Settings"):
        design_scale = st.slider("Design Size (%)", 10, 100, 50)
        offset_x = st.slider("Horizontal Offset (%)", 0, 100, 50)
        offset_y = st.slider("Vertical Offset (%)", 0, 100, 50)

    if dummy_file is not None and design_file is not None:
        if st.button("Generate Mockup"):
            with st.spinner("Generating Mockup..."):
                try:
                    output_image = generate_mockup(
                        dummy_file=dummy_file, 
                        design_file=design_file, 
                        bg_hex=bg_color, 
                        tshirt_hex=tshirt_color, 
                        blend_factor=blending_factor,
                        design_scale=design_scale,
                        offset_x=offset_x,
                        offset_y=offset_y
                    )
                    st.session_state['mockup_output'] = output_image
                    st.success("Mockup Generated!")
                except Exception as e:
                    st.error(f"Error during mockup generation: {e}")
                    st.session_state['mockup_output'] = None

    if 'mockup_output' in st.session_state and st.session_state['mockup_output'] is not None:
        st.subheader("Final Mockup Output")
        st.image(st.session_state['mockup_output'], use_column_width=True)
        
        st.download_button(
            label="Download Mockup PNG",
            data=convert_image_to_bytes(st.session_state['mockup_output']),
            file_name="mockup_output.png",
            mime="image/png"
        )

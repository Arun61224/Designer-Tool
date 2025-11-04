import streamlit as st
from PIL import Image, ImageChops, ImageDraw
import io
import os
import numpy as np

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def hex_to_rgb(hex_color):
    try:
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError):
        return (255, 255, 255)

def find_bounding_box(img_rgba):
    if img_rgba.mode != 'RGBA':
        img_rgba = img_rgba.convert('RGBA')
    
    alpha_channel = img_rgba.split()[-1]
    bbox = alpha_channel.getbbox()
    
    if bbox is None:
        return (0, 0, img_rgba.width, img_rgba.height)
    return bbox

def process_and_place_on_canvas(image_file, final_bg_color, tolerance, new_width, new_height, occupancy_percent, skip_bg_removal):
    
    img = Image.open(image_file)
    is_png = img.format == 'PNG'
    
    img_rgba = img.convert("RGBA")
    
    should_run_bg_removal = not is_png and not skip_bg_removal
    
    if should_run_bg_removal:
        img_rgb = img.convert("RGB")
        pixels_rgba = img_rgba.load()
        try:
            original_bg_color = img_rgb.getpixel((0, 0)) 
        except IndexError:
            st.error("Image size is too small for background detection.")
            return None

        # Call the dedicated BG removal logic for this canvas version
        img_rgba = _recursive_bg_removal(img_rgb, img_rgba, original_bg_color, tolerance) 
        pixels_rgba = img_rgba.load()


    left, upper, right, lower = find_bounding_box(img_rgba)
    object_cropped = img_rgba.crop((left, upper, right, lower))
    object_width = object_cropped.width
    object_height = object_cropped.height

    if object_width == 0 or object_height == 0:
        st.warning("Object area is zero after background processing. Outputting original image on canvas.")
        object_cropped = img_rgba

    target_max_dim = min(new_width, new_height)
    target_object_max_size = int(target_max_dim * (occupancy_percent / 100.0))

    aspect_ratio = object_cropped.width / object_cropped.height

    if aspect_ratio > 1:
        new_object_width = target_object_max_size
        new_object_height = int(new_object_width / aspect_ratio)
    else:
        new_object_height = target_object_max_size
        new_object_width = int(new_object_height * aspect_ratio)
    
    new_object_size = (new_object_width, new_object_height)
    
    resized_object = object_cropped.resize(new_object_size, Image.LANCZOS)

    final_bg_rgb = hex_to_rgb(final_bg_color)
    new_canvas = Image.new("RGB", (new_width, new_height), final_bg_rgb)
    
    x_offset = (new_width - new_object_width) // 2
    y_offset = (new_height - new_object_height) // 2
    
    new_canvas.paste(resized_object, (x_offset, y_offset), resized_object)

    return new_canvas

# New Helper Function for Recursive Background Removal (Flood Fill)
def _recursive_bg_removal(img_rgb, img_rgba, bg_color, tolerance):
    
    width, height = img_rgba.size
    pixels_rgba = img_rgba.load()
    pixels_rgb = img_rgb.load()
    
    # Use a set to store coordinates of pixels to check (Queue for BFS/Flood Fill)
    queue = [(0, 0)] # Start at top-left corner
    visited = set([(0, 0)])

    # Neighbors to check: left, right, up, down
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        x, y = queue.pop(0)

        # Check if the current pixel is close to the background color
        if color_distance(pixels_rgb[x, y], bg_color) < tolerance:
            # Make the current pixel transparent
            pixels_rgba[x, y] = (0, 0, 0, 0)
            
            # Check neighbors
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                
                # Check boundaries and if already visited
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    
                    # Only add to queue if the neighbor is also a background-like color
                    if color_distance(pixels_rgb[nx, ny], bg_color) < tolerance:
                         queue.append((nx, ny))
                         
    return img_rgba


def remove_background_to_png(image_file, tolerance):
    
    img = Image.open(image_file)
    
    img_rgba = img.convert("RGBA")
    img_rgb = img.convert("RGB")
    
    try:
        original_bg_color = img_rgb.getpixel((0, 0)) 
    except IndexError:
        return img_rgba

    # Use the new recursive function for better edge detection
    return _recursive_bg_removal(img_rgb, img_rgba, original_bg_color, tolerance)


def generate_mockup(dummy_file, design_file, bg_hex, tshirt_hex, blend_factor, design_scale, offset_x, offset_y, protected_hex_color_1=None, protected_hex_color_2=None):
    
    dummy_original = Image.open(dummy_file).convert("RGBA")
    design_original = Image.open(design_file).convert("RGBA")
    
    width, height = dummy_original.size

    bg_r, bg_g, bg_b = hex_to_rgb(bg_hex)
    final_img = Image.new('RGB', (width, height), (bg_r, bg_g, bg_b))
    
    tshirt_r, tshirt_g, tshirt_b = hex_to_rgb(tshirt_hex)
    tshirt_color_layer = Image.new('RGB', (width, height), (tshirt_r, tshirt_g, tshirt_b))
    
    blended_tshirt_rgb = Image.blend(dummy_original.convert('RGB'), tshirt_color_layer, blend_factor / 100.0)
    
    dummy_alpha = dummy_original.getchannel('A')
    dummy_rgb = dummy_original.convert('RGB')
    
    if protected_hex_color_1 or protected_hex_color_2:
        
        combined_protection_mask_array = np.zeros((height, width), dtype=np.uint8)
        
        protected_colors_hex = [protected_hex_color_1, protected_hex_color_2]
        PROTECTION_TOLERANCE = 25 
        
        pixels_dummy = dummy_rgb.load()
        
        for x in range(width):
            for y in range(height):
                if dummy_alpha.getpixel((x, y)) == 0:
                    continue

                current_pixel = pixels_dummy[x, y]
                
                is_protected = False
                for hex_color in protected_colors_hex:
                    if hex_color:
                        protected_rgb = hex_to_rgb(hex_color)
                        if color_distance(current_pixel, protected_rgb) < PROTECTION_TOLERANCE:
                            is_protected = True
                            break
                
                if is_protected:
                    combined_protection_mask_array[y, x] = 255
        
        protected_area_mask = Image.fromarray(combined_protection_mask_array, mode='L')
        
        blended_tshirt_rgb = Image.composite(
            dummy_rgb,
            blended_tshirt_rgb,
            protected_area_mask
        )

        final_img.paste(blended_tshirt_rgb, (0, 0), dummy_alpha)
    
    else:
        dummy_mask = dummy_original.getchannel('A')
        final_img.paste(blended_tshirt_rgb, (0, 0), dummy_mask)
    
    if design_file.getvalue():
        
        scale = design_scale / 100
        design_width = int(width * scale * 0.5) 
        aspect_ratio = design_original.height / design_original.width
        design_height = int(design_width * aspect_ratio)
        design_resized = design_original.resize((design_width, design_height), Image.LANCZOS)
        
        max_offset_x = width - design_width
        max_offset_y = height - design_height
        pos_x = int((offset_x / 100) * max_offset_x)
        pos_y = int((offset_y / 100) * max_offset_y)

        design_canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        design_canvas.paste(design_resized, (pos_x, pos_y), design_resized) 

        design_rgb = design_canvas.convert('RGB')
        design_alpha = design_canvas.getchannel('A')
        
        final_img.paste(design_rgb, (0, 0), design_alpha)

    return final_img

def convert_image_to_bytes(img):
    buf = io.BytesIO()
    # PNG is a lossless format, which maintains original quality
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

st.set_page_config(layout="wide", page_title="Image Processing Hub")

st.title("👕 Image Processing Hub (Streamlit)")
st.write("---")

tool_selection = st.sidebar.radio(
    "Select Tool",
    ("1. Background Remover / Canvas", "2. Design Mockup Tool", "3. Dedicated BG Remover (PNG Output)")
)

if tool_selection == "1. Background Remover / Canvas":
    st.header("🖼️ Background Remover / Canvas Processor")
    st.write("Upload an image to remove its background (based on top-left color) and place it on a new canvas.")
    st.markdown("💡 **Pro Tip:** PNG files (with transparent backgrounds) skip the background removal step automatically.")

    uploaded_file = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    # Logic to set default canvas size to original image dimensions for quality
    if uploaded_file is not None:
        if 'original_width' not in st.session_state or st.session_state.get('last_file_name') != uploaded_file.name:
            original_img = Image.open(uploaded_file)
            st.session_state['original_width'] = original_img.width
            st.session_state['original_height'] = original_img.height
            st.session_state['last_file_name'] = uploaded_file.name
            st.session_state['canvas_width_default'] = original_img.width
            st.session_state['canvas_height_default'] = original_img.height
            
            if 'canvas_width_default' not in st.session_state:
                 st.session_state['canvas_width_default'] = 1200
                 st.session_state['canvas_height_default'] = 1200
        
        default_w = st.session_state.get('canvas_width_default', 1200)
        default_h = st.session_state.get('canvas_height_default', 1200)
        
        st.markdown(f"**Original Resolution:** {default_w}x{default_h} px")
    else:
        default_w = 1200
        default_h = 1200

    with st.sidebar.expander("Canvas Settings"):
        canvas_width = st.number_input("Canvas Width (px)", value=default_w, min_value=100)
        canvas_height = st.number_input("Canvas Height (px)", value=default_h, min_value=100)
        occupancy = st.slider("Object Occupancy (%)", 50, 100, 95)
        bg_color = st.color_picker("Final Background Color", "#FFFFFF")
        
    with st.sidebar.expander("BG Removal Settings"):
        tolerance = st.slider("BG Removal Tolerance (0-200)", 0, 200, 30)
        skip_bg_removal = st.checkbox("Skip Color-Based Background Removal", value=False)
        st.caption("PNG files always skip this step.")
    
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
                col2.image(output_image, width=output_image.width if output_image else None) 
                
                st.download_button(
                    label="Download Processed PNG",
                    data=convert_image_to_bytes(output_image),
                    file_name="canvas_output.png",
                    mime="image/png"
                )

elif tool_selection == "2. Design Mockup Tool":
    st.header("👕 Design Mockup Tool (T-shirt/Product)")
    st.write("Upload a product image (like a t-shirt, PNG) and a design/logo to create a colored mockup.")
    st.markdown("⚠️ **WARNING:** Internal white colors in the **Design Image** will not be changed.")
    
    col_files = st.columns(2)
    dummy_file = col_files[0].file_uploader("1. Upload Product/Dummy Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="dummy_upload")
    design_file = col_files[1].file_uploader("2. Upload Separate Design/Logo (PNG, Optional)", type=["png"], key="design_upload")

    with st.sidebar.expander("Color & Blending Settings"):
        bg_color = st.color_picker("Background Color", "#FFFFFF", key="mockup_bg")
        tshirt_color = st.color_picker("Product Color Change To", "#0000FF", key="mockup_tshirt")
        blending_factor = st.slider("T-shirt Blending Factor (%)", 10, 100, 70)
        st.markdown("---")
        st.subheader("Internal Print Protection (Keep Print Color Safe)")
        protected_color_1 = st.color_picker("1st Color to Protect (e.g., White)", "#FFFFFF", key="protected_color_1")
        protected_color_2 = st.color_picker("2nd Color to Protect (e.g., Black/Red)", "#000000", key="protected_color_2")
        st.caption("Select the two main colors of the print (e.g., white and black outline) to prevent them from changing.")
        
    with st.sidebar.expander("Design Placement Settings"):
        design_scale = st.slider("Design Size (%)", 10, 100, 50)
        offset_x = st.slider("Horizontal Offset (%)", 0, 100, 50)
        offset_y = st.slider("Vertical Offset (%)", 0, 100, 50)

    if dummy_file is not None:
        if st.button("Generate Mockup"):
            with st.spinner("Generating Mockup..."):
                try:
                    design_input = design_file if design_file else io.BytesIO(b'')
                    
                    output_image = generate_mockup(
                        dummy_file=dummy_file, 
                        design_file=design_input,
                        bg_hex=bg_color, 
                        tshirt_hex=tshirt_color, 
                        blend_factor=blending_factor,
                        design_scale=design_scale,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        protected_hex_color_1=protected_color_1,
                        protected_hex_color_2=protected_color_2
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
        
elif tool_selection == "3. Dedicated BG Remover (PNG Output)":
    st.header("✨ Dedicated Background Remover (PNG Output)")
    st.write("Upload an image (JPG/PNG). The background will be removed (based on the top-left corner color) and the output will be saved as a high-quality transparent PNG.")

    uploaded_file = st.file_uploader("Upload Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="bg_remover_dedicated_upload")

    with st.sidebar.expander("Removal Settings"):
        tolerance = st.slider("BG Removal Tolerance (0-200)", 0, 200, 30, key="bg_remover_dedicated_tolerance")
    
    if uploaded_file is not None:
        if st.button("Remove Background & Convert to PNG"):
            with st.spinner("Processing..."):
                output_image = remove_background_to_png(
                    image_file=uploaded_file,
                    tolerance=tolerance
                )
            
            if output_image:
                st.success("Background Removal Complete!")
                
                col1, col2 = st.columns(2)
                col1.subheader("Original Image")
                col1.image(Image.open(uploaded_file), use_column_width=True)
                
                col2.subheader("Transparent PNG Output")
                col2.image(output_image, use_column_width=True)
                
                # The PNG format is lossless, ensuring maximum original quality is maintained.
                st.download_button(
                    label="Download Transparent PNG",
                    data=convert_image_to_bytes(output_image),
                    file_name="transparent_output.png",
                    mime="image/png"
                )

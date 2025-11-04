import streamlit as st
from PIL import Image, ImageChops, ImageDraw
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

def generate_mockup(dummy_file, design_file, bg_hex, tshirt_hex, blend_factor, design_scale, offset_x, offset_y, protected_hex_color=None):
    """
    Design Mockup logic (from app with background change.py)
    
    FIX: It isolates the design area and prevents color blending on the design itself.
    This works best if the dummy image is purely the product (like a blank t-shirt).
    If the design is baked into the dummy image, the user needs to upload a clean dummy.
    """
    
    # 1. Load Images
    dummy_original = Image.open(dummy_file).convert("RGBA")
    design_original = Image.open(design_file).convert("RGBA")
    
    width, height = dummy_original.size

    # 2. Background Canvas Setup
    bg_r, bg_g, bg_b = hex_to_rgb(bg_hex)
    final_img = Image.new('RGB', (width, height), (bg_r, bg_g, bg_b))
    
    # 3. Create Colored T-shirt Mask
    tshirt_r, tshirt_g, tshirt_b = hex_to_rgb(tshirt_hex)
    tshirt_color_layer = Image.new('RGB', (width, height), (tshirt_r, tshirt_g, tshirt_b))
    
    # Blend the T-shirt color with the original dummy image to retain shadows/folds
    # This blended image now has the new color but retains texture.
    blended_tshirt_rgb = Image.blend(dummy_original.convert('RGB'), tshirt_color_layer, blend_factor / 100.0)
    
    # --- NEW LOGIC: Protect Internal Print Color (Refined) ---
    
    # 3.1. Get the mask for the entire product area (Alpha Channel)
    dummy_alpha = dummy_original.getchannel('A')
    
    # Initialize the mask that controls where the color change happens (white=change, black=protect)
    color_change_mask = dummy_alpha.copy() 
    
    if protected_hex_color:
        protected_rgb = hex_to_rgb(protected_hex_color)
        
        # 3.2. Create a mask for the protected color area in the DUMMY IMAGE
        protected_area_mask = Image.new('L', (width, height), 0)
        
        # 3.3. Iterate and mask the area based on the protected color
        dummy_rgb = dummy_original.convert('RGB')
        pixels_dummy = dummy_rgb.load()
        pixels_mask = protected_area_mask.load()
        
        # Use a small tolerance (e.g., 20) for color matching
        PROTECTION_TOLERANCE = 35 # Increased tolerance for better detection
        
        for x in range(width):
            for y in range(height):
                current_pixel = pixels_dummy[x, y]
                # Check if the pixel color is close to the protected color AND it's inside the product
                # Check against the full product alpha mask to avoid background hits
                if color_distance(current_pixel, protected_rgb) < PROTECTION_TOLERANCE and dummy_alpha.getpixel((x, y)) > 0:
                    # If it's the protected color, set the mask pixel to WHITE (255)
                    pixels_mask[x, y] = 255
        
        # 3.4. Refine the main color_change_mask: 
        # The area to be *colored* should be the total product area (dummy_alpha) 
        # MINUS the protected print area (pixels_mask).
        # We invert the protected_area_mask to get a 'change' mask, then mask it with the alpha.
        
        # Create an inverted mask: White where print IS NOT (i.e., where we want to change color)
        # And Black where print IS (i.e., where we want to protect)
        
        # Note: PIL's point operation is non-trivial for numpy-like boolean ops. 
        # Easier to convert to NumPy array for subtraction.
        alpha_array = np.array(dummy_alpha, dtype=np.float32) / 255.0
        protected_array = np.array(protected_area_mask, dtype=np.float32) / 255.0
        
        # Subtraction: (Full Product Mask) - (Protected Print Mask)
        # This gives a mask for the CLOTH ONLY
        cloth_only_array = np.maximum(0, alpha_array - protected_array)
        
        # Convert back to PIL mask image
        cloth_only_mask = Image.fromarray((cloth_only_array * 255).astype(np.uint8))
        
        # 3.5. Composite the Original Product (Dummy) over the Colored Product using the Print Mask
        # Where the print is (protected area), we use the original dummy color (A).
        # Where the cloth is (colored area), we use the blended color (B).
        blended_tshirt_rgb = Image.composite(
            dummy_rgb,              # A: Original (for print area)
            blended_tshirt_rgb,     # B: Colored (for cloth area)
            protected_area_mask     # Mask: White=Original (A), Black=Colored (B)
        )

        # 3.6. Now, use the CLOTH ONLY mask to paste the new color onto the final image.
        # This mask ensures that blending only happens on the cloth area.
        final_img.paste(blended_tshirt_rgb, (0, 0), dummy_mask)
    
    # --- END NEW LOGIC ---
    
    else:
        # If no protected color is provided, use the simple paste method from before
        dummy_mask = dummy_original.getchannel('A')
        final_img.paste(blended_tshirt_rgb, (0, 0), dummy_mask)

    # 4. Place and Blend Design (This logic is for when a design is uploaded separately)
    
    # Check if a separate design file was uploaded (design_file)
    if design_file.getvalue():
        # The logic below assumes the uploaded design is separate and should keep its color.
        
        # Calculate size and position for the design
        scale = design_scale / 100
        design_width = int(width * scale * 0.5) 
        aspect_ratio = design_original.height / design_original.width
        design_height = int(design_width * aspect_ratio)
        design_resized = design_original.resize((design_width, design_height), Image.LANCZOS)
        
        # Calculate offsets
        max_offset_x = width - design_width
        max_offset_y = height - design_height
        pos_x = int((offset_x / 100) * max_offset_x)
        pos_y = int((offset_y / 100) * max_offset_y)

        # Create a blank canvas for the design at the final size
        design_canvas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        design_canvas.paste(design_resized, (pos_x, pos_y), design_resized) 

        design_rgb = design_canvas.convert('RGB')
        design_alpha = design_canvas.getchannel('A')
        
        # Paste the design onto the mockup image
        final_img.paste(design_rgb, (0, 0), design_alpha)

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
    st.markdown("⚠️ **ज़रूरी सूचना (IMPORTANT):** **Design Image** के अंदर अगर कोई सफ़ेद रंग है, तो वह रंग नहीं बदलेगा।")
    
    col_files = st.columns(2)
    # NOTE: Design file is now optional for cases where the design is part of the dummy.
    dummy_file = col_files[0].file_uploader("1. Upload Product/Dummy Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="dummy_upload")
    design_file = col_files[1].file_uploader("2. Upload Separate Design/Logo (PNG, Optional)", type=["png"], key="design_upload")

    # Parameters in a sidebar for a clean look
    with st.sidebar.expander("Color & Blending Settings"):
        bg_color = st.color_picker("Background Color", "#FFFFFF", key="mockup_bg")
        tshirt_color = st.color_picker("Product Color Change To", "#0000FF", key="mockup_tshirt")
        blending_factor = st.slider("T-shirt Blending Factor (%)", 10, 100, 70)
        st.markdown("---")
        # NEW INPUT FOR PRINT PROTECTION
        st.subheader("Internal Print Protection")
        protected_color = st.color_picker("Color to Protect (Print Color)", "#FFFFFF", key="protected_color")
        st.caption("यह रंग (जैसे Print का White) बदलने से बचाएगा।")
        
    with st.sidebar.expander("Design Placement Settings"):
        design_scale = st.slider("Design Size (%)", 10, 100, 50)
        offset_x = st.slider("Horizontal Offset (%)", 0, 100, 50)
        offset_y = st.slider("Vertical Offset (%)", 0, 100, 50)

    if dummy_file is not None: # Design file is now optional
        if st.button("Generate Mockup"):
            with st.spinner("Generating Mockup..."):
                try:
                    # Check if the optional design file exists
                    design_input = design_file if design_file else io.BytesIO(b'')
                    
                    output_image = generate_mockup(
                        dummy_file=dummy_file, 
                        design_file=design_input, # Pass an empty BytesIO if no file is uploaded
                        bg_hex=bg_color, 
                        tshirt_hex=tshirt_color, 
                        blend_factor=blending_factor,
                        design_scale=design_scale,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        protected_hex_color=protected_color
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

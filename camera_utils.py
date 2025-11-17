import bpy
import math
import mathutils
from mathutils import Vector, Matrix
import cv2
import numpy as np
import tempfile
import os

def setup_camera_background(camera_obj, image, opacity=0.5):
    """Setup background image for camera viewport"""
    if camera_obj.type != 'CAMERA':
        return
    
    camera_data = camera_obj.data
    
    # Clear existing background images
    camera_data.background_images.clear()
    
    # Add new background image
    bg = camera_data.background_images.new()
    bg.image = image
    bg.alpha = opacity
    bg.display_depth = 'FRONT'
    bg.frame_method = 'FIT'
    bg.show_background_image = True
    bg.show_on_foreground = True
    
    # Enable background images display in viewport
    camera_data.show_background_images = True
    
    return bg

def get_camera_frustum_corners(camera_obj, distance=1.0):
    """Get the corners of camera frustum at given distance"""
    if camera_obj.type != 'CAMERA':
        return None
    
    camera = camera_obj.data
    
    # Get camera parameters
    if camera.type == 'PERSP':
        # Perspective camera
        aspect = camera.sensor_width / camera.sensor_height
        fov = camera.angle
        
        # Calculate frustum dimensions at distance
        height = 2.0 * distance * math.tan(fov / 2.0)
        width = height * aspect
    else:
        # Orthographic camera
        width = camera.ortho_scale
        height = width / (camera.sensor_width / camera.sensor_height)
    
    # Define corners in camera space
    half_w = width / 2.0
    half_h = height / 2.0
    
    corners = [
        Vector((-half_w, -half_h, -distance)),  # Bottom-left
        Vector((half_w, -half_h, -distance)),   # Bottom-right
        Vector((half_w, half_h, -distance)),    # Top-right
        Vector((-half_w, half_h, -distance)),   # Top-left
    ]
    
    # Transform to world space
    world_matrix = camera_obj.matrix_world
    world_corners = [world_matrix @ corner for corner in corners]
    
    return world_corners

def screen_to_camera_space(camera_obj, screen_x, screen_y, resolution_x, resolution_y):
    """Convert screen coordinates to camera frustum coordinates"""
    # Normalize screen coordinates to [-1, 1]
    ndc_x = (screen_x / resolution_x) * 2.0 - 1.0
    ndc_y = 1.0 - (screen_y / resolution_y) * 2.0  # Flip Y
    
    # Get aspect ratio
    camera = camera_obj.data
    aspect = camera.sensor_width / camera.sensor_height
    
    # Calculate position on frustum at distance 1.0
    if camera.type == 'PERSP':
        fov = camera.angle
        height = 2.0 * mathutils.math.tan(fov / 2.0)
        width = height * aspect
        
        x = ndc_x * width / 2.0
        y = ndc_y * height / 2.0
        z = -1.0
    else:
        width = camera.ortho_scale
        height = width / aspect
        
        x = ndc_x * width / 2.0
        y = ndc_y * height / 2.0
        z = -1.0
    
    # Transform to world space
    camera_space = Vector((x, y, z))
    world_pos = camera_obj.matrix_world @ camera_space
    
    return world_pos

def project_world_to_screen(camera_obj, world_pos, resolution_x, resolution_y):
    """Project world position to screen coordinates"""
    # Transform to camera space
    camera_matrix_inv = camera_obj.matrix_world.inverted()
    camera_space = camera_matrix_inv @ world_pos
    
    # Get camera parameters
    camera = camera_obj.data
    aspect = camera.sensor_width / camera.sensor_height
    
    if camera.type == 'PERSP':
        # Perspective projection
        if camera_space.z >= 0:  # Behind camera
            return None
        
        fov = camera.angle
        distance = -camera_space.z
        height = 2.0 * distance * math.tan(fov / 2.0)
        width = height * aspect
        
        ndc_x = camera_space.x / (width / 2.0)
        ndc_y = camera_space.y / (height / 2.0)
    else:
        # Orthographic projection
        width = camera.ortho_scale
        height = width / aspect
        
        ndc_x = camera_space.x / (width / 2.0)
        ndc_y = camera_space.y / (height / 2.0)
    
    # Convert NDC to screen space
    screen_x = (ndc_x + 1.0) / 2.0 * resolution_x
    screen_y = (1.0 - ndc_y) / 2.0 * resolution_y
    
    return (screen_x, screen_y)

def is_point_in_camera_view(camera_obj, world_pos):
    """Check if a world position is visible in camera view"""
    camera_matrix_inv = camera_obj.matrix_world.inverted()
    camera_space = camera_matrix_inv @ world_pos

    # Check if behind camera
    if camera_space.z >= 0:
        return False

    camera = camera_obj.data
    aspect = camera.sensor_width / camera.sensor_height

    if camera.type == 'PERSP':
        fov = camera.angle
        distance = -camera_space.z
        height = 2.0 * distance * math.tan(fov / 2.0)
        width = height * aspect

        half_w = width / 2.0
        half_h = height / 2.0
    else:
        width = camera.ortho_scale
        height = width / aspect
        half_w = width / 2.0
        half_h = height / 2.0

    # Check if within frustum bounds
    return (abs(camera_space.x) <= half_w and
            abs(camera_space.y) <= half_h)

def get_camera_render_resolution(scene):
    """Get the camera render resolution with percentage applied"""
    render_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    render_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    return render_width, render_height

def fit_image_to_camera(image_path, camera_obj, scene):
    """
    Fit input image to camera render resolution maintaining aspect ratio.
    Returns: (fitted_opencv_image, original_opencv_image, fit_info)

    fit_info contains:
        - render_width, render_height: camera resolution
        - original_width, original_height: original image size
        - fitted_width, fitted_height: size of image within camera frame
        - offset_x, offset_y: black bar offsets
        - scale_factor: scaling from original to fitted size
    """
    # Load original image with OpenCV
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_height, orig_width = original_img.shape[:2]

    # Get camera render resolution
    render_width, render_height = get_camera_render_resolution(scene)

    # Calculate aspect ratios
    render_aspect = render_width / render_height
    image_aspect = orig_width / orig_height

    # Calculate how image fits in camera (FIT mode - maintain aspect ratio)
    if image_aspect > render_aspect:
        # Image is wider - fit to width, add letterbox (top/bottom bars)
        fitted_width = render_width
        fitted_height = int(render_width / image_aspect)
        offset_x = 0
        offset_y = (render_height - fitted_height) // 2
        scale_factor = render_width / orig_width
    else:
        # Image is taller or same - fit to height, add pillarbox (left/right bars)
        fitted_height = render_height
        fitted_width = int(render_height * image_aspect)
        offset_x = (render_width - fitted_width) // 2
        offset_y = 0
        scale_factor = render_height / orig_height

    # Create fitted image with black bars
    fitted_img = np.zeros((render_height, render_width, 3), dtype=np.uint8)

    # Resize original image to fitted size
    resized_img = cv2.resize(original_img, (fitted_width, fitted_height),
                             interpolation=cv2.INTER_LINEAR)

    # Place resized image in the fitted frame
    fitted_img[offset_y:offset_y+fitted_height, offset_x:offset_x+fitted_width] = resized_img

    # Prepare fit info
    fit_info = {
        'render_width': render_width,
        'render_height': render_height,
        'original_width': orig_width,
        'original_height': orig_height,
        'fitted_width': fitted_width,
        'fitted_height': fitted_height,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'scale_factor': scale_factor
    }

    print(f"\n=== Image Fitting Info ===")
    print(f"Original: {orig_width}x{orig_height}")
    print(f"Camera render: {render_width}x{render_height}")
    print(f"Fitted: {fitted_width}x{fitted_height}")
    print(f"Offset: ({offset_x}, {offset_y})")
    print(f"Scale factor: {scale_factor:.4f}")
    print("=" * 50)

    return fitted_img, original_img, fit_info

def save_opencv_image_as_temp(opencv_image, prefix="temp_camera_bg"):
    """
    Save OpenCV image to a temporary file and return the path.
    The file will be in the system temp directory.
    """
    temp_dir = tempfile.gettempdir()
    temp_filename = f"{prefix}_{os.getpid()}.png"
    temp_path = os.path.join(temp_dir, temp_filename)

    cv2.imwrite(temp_path, opencv_image)
    print(f"Saved temporary image: {temp_path}")

    return temp_path

def cleanup_temp_image(temp_path):
    """Remove temporary image file"""
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
            print(f"Cleaned up temporary image: {temp_path}")
        except Exception as e:
            print(f"Could not delete temp file {temp_path}: {e}")

def map_fitted_coords_to_original(point, fit_info):
    """
    Map coordinates from fitted image space back to original image space.

    Args:
        point: (x, y) in fitted image coordinates
        fit_info: dictionary from fit_image_to_camera

    Returns:
        (x, y) in original image coordinates, or None if outside fitted area
    """
    x, y = point

    # Remove offset (black bars)
    x_in_fitted = x - fit_info['offset_x']
    y_in_fitted = y - fit_info['offset_y']

    # Check if within fitted area
    if (x_in_fitted < 0 or x_in_fitted >= fit_info['fitted_width'] or
        y_in_fitted < 0 or y_in_fitted >= fit_info['fitted_height']):
        return None

    # Scale back to original image size
    orig_x = int(x_in_fitted / fit_info['scale_factor'])
    orig_y = int(y_in_fitted / fit_info['scale_factor'])

    # Clamp to original image bounds
    orig_x = max(0, min(orig_x, fit_info['original_width'] - 1))
    orig_y = max(0, min(orig_y, fit_info['original_height'] - 1))

    return (orig_x, orig_y)

def map_original_coords_to_fitted(point, fit_info):
    """
    Map coordinates from original image space to fitted image space.

    Args:
        point: (x, y) in original image coordinates
        fit_info: dictionary from fit_image_to_camera

    Returns:
        (x, y) in fitted image coordinates
    """
    x, y = point

    # Scale to fitted size
    fitted_x = int(x * fit_info['scale_factor'])
    fitted_y = int(y * fit_info['scale_factor'])

    # Add offset
    final_x = fitted_x + fit_info['offset_x']
    final_y = fitted_y + fit_info['offset_y']

    return (final_x, final_y)
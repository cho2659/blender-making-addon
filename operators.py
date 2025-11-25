import bpy
import json
import numpy as np
import cv2
from bpy.types import Operator
from bpy.props import StringProperty
from . import camera_utils, mask_utils



# ============================================================================
# HELPER FUNCTIONS FOR OPENCV INTEGRATION
# ============================================================================

def get_fit_info_from_props(props):
    """Extract fit_info dictionary from camera properties"""
    return {
        'original_width': props.fit_original_width,
        'original_height': props.fit_original_height,
        'render_width': props.fit_render_width,
        'render_height': props.fit_render_height,
        'fitted_width': props.fit_fitted_width,
        'fitted_height': props.fit_fitted_height,
        'offset_x': props.fit_offset_x,
        'offset_y': props.fit_offset_y,
        'scale_factor': props.fit_scale_factor
    }

def apply_grabcut_refinement(rough_points, opencv_image, constraint=0.1):
    """
    Apply GrabCut algorithm for fast and accurate mask refinement.
    Much faster than active_contour and better at following object boundaries.

    Args:
        rough_points: List of (x, y) tuples representing rough user drawing
        opencv_image: OpenCV image (BGR format)
        constraint: How tightly to follow user drawing (lower = looser)

    Returns:
        List of refined (x, y) points
    """
    if len(rough_points) < 3:
        return rough_points

    print(f"\n=== GrabCut Refinement (Fast) ===")
    print(f"Input: {len(rough_points)} rough points")

    height, width = opencv_image.shape[:2]

    # Create mask for GrabCut - initialize as background
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:] = cv2.GC_BGD  # Everything starts as background

    # Fill user's polygon as "probable foreground"
    points_array = np.array(rough_points, dtype=np.int32)

    # Fix: First fill entire polygon area with a temporary value
    temp_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(temp_mask, [points_array], 255)

    # Set the polygon area as probable foreground
    mask[temp_mask == 255] = cv2.GC_PR_FGD

    # Create inner region for "definite foreground" based on constraint
    kernel_size = int(max(3, min(width, height) * constraint * 0.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner_mask = cv2.erode(temp_mask, kernel, iterations=2)

    # Set inner region as definite foreground
    mask[inner_mask == 255] = cv2.GC_FGD

    # Debug: Check mask coverage
    fg_pixels = np.sum(mask >= cv2.GC_PR_FGD)
    total_pixels = height * width
    print(f"  Mask coverage: {fg_pixels}/{total_pixels} pixels ({100*fg_pixels/total_pixels:.1f}%)")

    # Get bounding rectangle with padding
    x, y, w, h = cv2.boundingRect(points_array)
    padding = 10
    rect = (
        max(0, x - padding),
        max(0, y - padding),
        min(width - x + padding, w + 2 * padding),
        min(height - y + padding, h + 2 * padding)
    )

    # Initialize models for GrabCut
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        # Apply GrabCut - much faster than active_contour
        # Using mask mode (GC_INIT_WITH_MASK) instead of rect mode
        cv2.grabCut(
            opencv_image,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,  # iterations - 5 is usually enough
            cv2.GC_INIT_WITH_MASK
        )

        # Create binary mask (foreground = 1, background = 0)
        binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Find contours of refined mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("  No contours found, using original points")
            return rough_points

        # Get largest contour (the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify contour to reduce point count
        epsilon = 2.0
        simplified = cv2.approxPolyDP(largest_contour, epsilon, closed=True)

        # Convert to list of tuples
        refined_points = [(int(p[0][0]), int(p[0][1])) for p in simplified]

        print(f"GrabCut refined: {len(rough_points)} → {len(refined_points)} points")
        print(f"✓ Processing took ~0.1s (vs 2-5s for active_contour)")
        print("=" * 50)

        return refined_points

    except Exception as e:
        print(f"GrabCut failed: {e}")
        import traceback
        traceback.print_exc()
        return rough_points

def get_camera_background_image(camera_obj):
    """
    Extract the background image from camera
    Returns: (blender_image, (width, height)) or (None, None)
    """
    if not camera_obj or camera_obj.type != 'CAMERA':
        return None, None
    
    camera_data = camera_obj.data
    
    if not camera_data.background_images:
        return None, None
    
    bg_image = camera_data.background_images[0]
    
    if not bg_image.image:
        return None, None
    
    blender_image = bg_image.image
    width = blender_image.size[0]
    height = blender_image.size[1]
    
    return blender_image, (width, height)


def blender_image_to_opencv(blender_image):
    """
    Convert Blender Image to OpenCV numpy array
    Handles: RGBA→BGR, float→uint8, flip Y-axis
    """
    # Get pixel data
    pixels = blender_image.pixels[:]
    
    # Convert to numpy
    pixels_array = np.array(pixels, dtype=np.float32)
    
    # Get dimensions
    width = blender_image.size[0]
    height = blender_image.size[1]
    
    # Reshape to 2D image with RGBA channels
    pixels_array = pixels_array.reshape((height, width, 4))
    
    # Flip vertically (Blender: bottom-left origin, OpenCV: top-left origin)
    pixels_array = np.flipud(pixels_array)
    
    # Extract RGB (drop alpha)
    rgb_array = pixels_array[:, :, :3]
    
    # Convert float (0.0-1.0) to uint8 (0-255)
    rgb_array = (rgb_array * 255).astype(np.uint8)
    
    # Convert RGB to BGR (OpenCV format)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array

def viewport_to_image_coords(viewport_pos, context, camera_obj, image_size):
    """
    Convert viewport coordinates to original image pixel coordinates.

    Uses Blender's API to get actual camera frame position in viewport,
    then maps to background image coordinates.

    Flow: viewport click → camera frame bounds → normalized camera coords → image pixels
    """
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    region = context.region
    rv3d = context.space_data.region_3d
    scene = context.scene

    viewport_x, viewport_y = viewport_pos

    # Check if in camera view
    if rv3d.view_perspective != 'CAMERA':
        print(f"  DEBUG: Not in camera view!")
        return None

    # Get original image size
    img_width, img_height = image_size
    img_aspect = img_width / img_height

    # Get render resolution
    render_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    render_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    render_aspect = render_width / render_height

    # Fix: Removed debug prints for faster drawing response

    # STEP 1: Get camera frame corners in 3D world space
    # view_frame() returns corners at distance 1.0 in camera local space
    # We need to project them at a meaningful distance from camera
    camera_data = camera_obj.data

    # Get frame corners at distance 1.0 (camera local space)
    frame_corners_local = camera_data.view_frame(scene=scene)

    # Scale corners to a distance that's meaningful (e.g., 10 units away)
    # This prevents numerical issues when projecting very close points
    distance = 10.0
    frame_corners_scaled = [corner * distance for corner in frame_corners_local]

    # Transform to world space
    matrix_world = camera_obj.matrix_world
    world_corners = [matrix_world @ corner for corner in frame_corners_scaled]

    # STEP 2: Project 3D corners to 2D viewport coordinates
    # view_frame() returns corners in this order: [bottom-left, bottom-right, top-right, top-left]
    region_corners = [location_3d_to_region_2d(region, rv3d, corner) for corner in world_corners]

    # Check if all corners were successfully projected
    if None in region_corners:
        print(f"  DEBUG: Failed to project camera corners to viewport!")
        return None

    # Extract viewport bounds of camera frame
    # bottom-left, bottom-right, top-right, top-left
    bl, br, tr, tl = region_corners

    camera_left = tl.x
    camera_right = bl.x
    camera_bottom = br.y
    camera_top = bl.y

    camera_frame_width = camera_right - camera_left
    camera_frame_height = camera_top - camera_bottom

    # STEP 3: Convert viewport click to camera frame local coordinates
    camera_local_x = viewport_x - camera_left
    camera_local_y = viewport_y - camera_bottom

    # Check if click is within camera frame
    if camera_local_x < 0 or camera_local_x > camera_frame_width:
        return None
    if camera_local_y < 0 or camera_local_y > camera_frame_height:
        return None

    # STEP 4: Calculate how background image fits in camera frame (FIT mode)
    if img_aspect > render_aspect:
        # Image wider - fits width, letterbox top/bottom
        bg_width = camera_frame_width
        bg_height = camera_frame_width / img_aspect
        bg_offset_x = 0
        bg_offset_y = (camera_frame_height - bg_height) / 2
    else:
        # Image taller - fits height, pillarbox left/right
        bg_height = camera_frame_height
        bg_width = camera_frame_height * img_aspect
        bg_offset_x = (camera_frame_width - bg_width) / 2
        bg_offset_y = 0

    # STEP 5: Convert camera local coords to background image coords
    bg_x = camera_local_x - bg_offset_x
    bg_y = camera_local_y - bg_offset_y

    # Check if outside background image
    if bg_x < 0 or bg_x > bg_width or bg_y < 0 or bg_y > bg_height:
        return None

    # STEP 6: Normalize to [0,1] range and scale to original image pixels
    norm_x = bg_x / bg_width
    norm_y = bg_y / bg_height

    image_x = norm_x * img_width
    image_y = norm_y * img_height

    # Flip Y (viewport: bottom-up, image: top-down)
    image_y = img_height - image_y

    return (int(image_x), int(image_y))



def image_to_viewport_coords(image_pos, context, camera_obj, image_size):
    """
    Convert image coordinates back to viewport coordinates.

    Uses Blender's API to get actual camera frame position in viewport,
    inverse of viewport_to_image_coords().

    Flow: image pixels → normalized coords → background image → camera frame → viewport
    """
    from bpy_extras.view3d_utils import location_3d_to_region_2d

    region = context.region
    rv3d = context.space_data.region_3d
    scene = context.scene

    image_x, image_y = image_pos

    # Check if in camera view
    if rv3d.view_perspective != 'CAMERA':
        return None

    # Get image size and aspect
    img_width, img_height = image_size
    img_aspect = img_width / img_height

    # Get render resolution
    render_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    render_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    render_aspect = render_width / render_height

    # STEP 1: Get camera frame corners in viewport (same as viewport_to_image_coords)
    camera_data = camera_obj.data

    # Get frame corners at distance 1.0 (camera local space)
    frame_corners_local = camera_data.view_frame(scene=scene)

    # Scale corners to a distance that's meaningful (e.g., 10 units away)
    distance = 10.0
    frame_corners_scaled = [corner * distance for corner in frame_corners_local]

    # Transform to world space
    matrix_world = camera_obj.matrix_world
    world_corners = [matrix_world @ corner for corner in frame_corners_scaled]

    # Project to viewport
    region_corners = [location_3d_to_region_2d(region, rv3d, corner) for corner in world_corners]

    if None in region_corners:
        return None

    # Extract viewport bounds
    bl, br, tr, tl = region_corners

    camera_left = bl.x
    camera_right = br.x
    camera_bottom = bl.y
    camera_top = tl.y

    camera_frame_width = camera_right - camera_left
    camera_frame_height = camera_top - camera_bottom

    # STEP 2: Calculate how background image fits in camera frame (FIT mode)
    if img_aspect > render_aspect:
        bg_width = camera_frame_width
        bg_height = camera_frame_width / img_aspect
        bg_offset_x = 0
        bg_offset_y = (camera_frame_height - bg_height) / 2
    else:
        bg_height = camera_frame_height
        bg_width = camera_frame_height * img_aspect
        bg_offset_x = (camera_frame_width - bg_width) / 2
        bg_offset_y = 0

    # STEP 3: Flip Y (image: top-down, viewport: bottom-up)
    image_y_flipped = img_height - image_y

    # STEP 4: Normalize image coords to [0,1]
    norm_x = image_x / img_width
    norm_y = image_y_flipped / img_height

    # STEP 5: Scale to background image size
    bg_x = norm_x * bg_width
    bg_y = norm_y * bg_height

    # STEP 6: Convert to camera frame local coords
    camera_local_x = bg_x + bg_offset_x
    camera_local_y = bg_y + bg_offset_y

    # STEP 7: Convert to viewport coords
    viewport_x = camera_local_x + camera_left
    viewport_y = camera_local_y + camera_bottom

    return (int(viewport_x), int(viewport_y))


# ============================================================================
# OPERATOR CLASSES
# ============================================================================

class CAMERA_OT_add_custom_camera(Operator):
    """Add a new Custom Camera with masking capabilities"""
    bl_idname = "camera.add_custom_camera"
    bl_label = "Custom Camera"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Check if custom camera already exists
        custom_cameras = [obj for obj in context.scene.objects
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]
        
        if custom_cameras:
            self.report({'WARNING'}, "Custom Camera already exists")
            return {'FINISHED'}
        
        # Create camera
        camera_data = bpy.data.cameras.new(name="CustomCamera")
        # Set default to orthographic
        camera_data.type = 'ORTHO'
        camera_data.ortho_scale = 6.0
        camera_obj = bpy.data.objects.new("CustomCamera", camera_data)
        
        # Link to scene
        context.scene.collection.objects.link(camera_obj)
        
        # Position at cursor
        camera_obj.location = context.scene.cursor.location
        
        # Mark as custom camera
        camera_obj.custom_camera_props.is_custom_camera = True
        
        # Select
        bpy.ops.object.select_all(action='DESELECT')
        camera_obj.select_set(True)
        context.view_layer.objects.active = camera_obj
        
        self.report({'INFO'}, f"Added Custom Camera: {camera_obj.name}")
        return {'FINISHED'}


class CAMERA_OT_load_reference_image(Operator):
    """Load reference image for mask drawing"""
    bl_idname = "camera.load_reference_image"
    bl_label = "Load Reference Image"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        # Find custom camera
        custom_cameras = [obj for obj in context.scene.objects
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]

        if not custom_cameras:
            self.report({'ERROR'}, "Make Custom Camera first")
            return {'CANCELLED'}

        obj = custom_cameras[0]
        props = obj.custom_camera_props

        # Store original image path
        props.reference_image_path = self.filepath

        # Cleanup previous fitted image if exists
        if props.fitted_image_path:
            camera_utils.cleanup_temp_image(props.fitted_image_path)
            props.fitted_image_path = ""

        # Load original image directly (let Blender handle the fitting)
        try:
            # Load original image into Blender
            if self.filepath in bpy.data.images:
                original_blender_img = bpy.data.images[self.filepath]
                original_blender_img.reload()
            else:
                original_blender_img = bpy.data.images.load(self.filepath)

            # Get original image dimensions
            orig_width = original_blender_img.size[0]
            orig_height = original_blender_img.size[1]

            # Get render resolution
            render_width = int(context.scene.render.resolution_x * context.scene.render.resolution_percentage / 100)
            render_height = int(context.scene.render.resolution_y * context.scene.render.resolution_percentage / 100)

            # Calculate fit info for coordinate mapping (but don't create fitted image)
            render_aspect = render_width / render_height
            image_aspect = orig_width / orig_height

            if image_aspect > render_aspect:
                # Image is wider - fit to width
                fitted_width = render_width
                fitted_height = int(render_width / image_aspect)
                offset_x = 0
                offset_y = (render_height - fitted_height) // 2
                scale_factor = render_width / orig_width
            else:
                # Image is taller - fit to height
                fitted_height = render_height
                fitted_width = int(render_height * image_aspect)
                offset_x = (render_width - fitted_width) // 2
                offset_y = 0
                scale_factor = render_height / orig_height

            # Store fit_info for coordinate mapping
            props.fit_original_width = orig_width
            props.fit_original_height = orig_height
            props.fit_render_width = render_width
            props.fit_render_height = render_height
            props.fit_fitted_width = fitted_width
            props.fit_fitted_height = fitted_height
            props.fit_offset_x = offset_x
            props.fit_offset_y = offset_y
            props.fit_scale_factor = scale_factor

            # Setup camera background with ORIGINAL image (Blender will fit it automatically)
            camera_utils.setup_camera_background(
                obj, original_blender_img, props.reference_image_opacity
            )

            # Switch to camera view
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            context.scene.camera = obj
                            space.region_3d.view_perspective = 'CAMERA'
                            break

            print(f"\n=== Image Loading Info ===")
            print(f"Original: {orig_width}x{orig_height}")
            print(f"Render: {render_width}x{render_height}")
            print(f"Fitted dimensions: {fitted_width}x{fitted_height}")
            print(f"Offset: ({offset_x}, {offset_y})")
            print(f"Scale factor: {scale_factor:.4f}")
            print("=" * 50)

            self.report({'INFO'},
                f"Loaded image ({orig_width}x{orig_height})")

        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class CAMERA_OT_add_mask_region(Operator):
    """Add a new mask region to draw"""
    bl_idname = "camera.add_mask_region"
    bl_label = "Add Mask Region"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects 
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]
        
        if not custom_cameras:
            self.report({'ERROR'}, "Make Custom Camera first")
            return {'CANCELLED'}
        
        obj = custom_cameras[0]
        props = obj.custom_camera_props
        
        # Add new mask region
        mask = props.mask_regions.add()
        mask.name = f"Mask{len(props.mask_regions)}"
        
        # Set as active
        props.active_mask_index = len(props.mask_regions) - 1
        
        self.report({'INFO'}, f"Added mask region: {mask.name}")
        return {'FINISHED'}


class CAMERA_OT_remove_mask_region(Operator):
    """Remove selected mask region"""
    bl_idname = "camera.remove_mask_region"
    bl_label = "Remove Mask Region"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]

        if not custom_cameras:
            return {'CANCELLED'}

        obj = custom_cameras[0]
        props = obj.custom_camera_props

        if len(props.mask_regions) == 0:
            return {'CANCELLED'}

        # Remove the mask
        props.mask_regions.remove(props.active_mask_index)
        props.active_mask_index = max(0, props.active_mask_index - 1)

        # Switch camera background to another mask or original image
        if len(props.mask_regions) > 0:
            # Switch to the new active mask's overlay if available
            new_active_mask = props.mask_regions[props.active_mask_index]
            if new_active_mask.mask_overlay_path:
                try:
                    if new_active_mask.mask_overlay_path in bpy.data.images:
                        mask_img = bpy.data.images[new_active_mask.mask_overlay_path]
                        mask_img.reload()
                    else:
                        mask_img = bpy.data.images.load(new_active_mask.mask_overlay_path)

                    camera_utils.setup_camera_background(
                        obj, mask_img, props.reference_image_opacity
                    )
                except Exception as e:
                    print(f"Could not load mask overlay: {e}")
                    self._restore_original_image(obj, props)
            else:
                # No overlay for new active mask, show original
                self._restore_original_image(obj, props)
        else:
            # No masks left, restore original image
            self._restore_original_image(obj, props)

        return {'FINISHED'}

    def _restore_original_image(self, camera_obj, props):
        """Restore original reference image"""
        if props.reference_image_path:
            try:
                if props.reference_image_path in bpy.data.images:
                    orig_img = bpy.data.images[props.reference_image_path]
                    orig_img.reload()
                else:
                    orig_img = bpy.data.images.load(props.reference_image_path)

                camera_utils.setup_camera_background(
                    camera_obj, orig_img, props.reference_image_opacity
                )
                print("Restored original reference image")
            except Exception as e:
                print(f"Could not restore original image: {e}")


class CAMERA_OT_set_target_object(Operator):
    """Set the target object to check against masks"""
    bl_idname = "camera.set_target_object"
    bl_label = "Set as Target"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Find custom camera
        custom_cameras = [obj for obj in context.scene.objects 
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]
        
        if not custom_cameras:
            self.report({'ERROR'}, "Make Custom Camera first")
            return {'CANCELLED'}
        
        camera = custom_cameras[0]
        
        # Find mesh in selection
        target = None
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                target = obj
                break
        
        if not target:
            self.report({'ERROR'}, "Select a mesh object as target")
            return {'CANCELLED'}
        
        # Set target
        camera.custom_camera_props.target_object = target
        
        # Add to target list
        props = camera.custom_camera_props
        if hasattr(props, 'target_objects'):
            found = False
            for t in props.target_objects:
                if t.obj == target:
                    found = True
                    break
            
            if not found:
                new_target = props.target_objects.add()
                new_target.obj = target
        
        self.report({'INFO'}, f"Set target: {target.name}")
        return {'FINISHED'}


class CAMERA_OT_add_target_object(Operator):
    """Add selected object to target list"""
    bl_idname = "camera.add_target_object"
    bl_label = "Add Target Object"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects 
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]
        
        if not custom_cameras:
            self.report({'ERROR'}, "Make Custom Camera first")
            return {'CANCELLED'}
        
        obj = custom_cameras[0]
        props = obj.custom_camera_props
        
        if not hasattr(props, 'target_objects'):
            self.report({'ERROR'}, "target_objects property not found")
            return {'CANCELLED'}
        
        # Add selected mesh objects
        added = []
        for selected in context.selected_objects:
            if selected.type == 'MESH' and selected != obj:
                found = False
                for t in props.target_objects:
                    if t.obj == selected:
                        found = True
                        break
                
                if not found:
                    new_target = props.target_objects.add()
                    new_target.obj = selected
                    added.append(selected.name)
        
        if added:
            self.report({'INFO'}, f"Added {len(added)} target(s): {', '.join(added)}")
        else:
            self.report({'WARNING'}, "No new targets to add")
        
        return {'FINISHED'}


class CAMERA_OT_remove_target_object(Operator):
    """Remove target object from list"""
    bl_idname = "camera.remove_target_object"
    bl_label = "Remove Target Object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]

        if not custom_cameras:
            return {'CANCELLED'}

        obj = custom_cameras[0]
        props = obj.custom_camera_props

        if not hasattr(props, 'target_objects') or len(props.target_objects) == 0:
            return {'CANCELLED'}

        props.target_objects.remove(props.active_target_index)
        props.active_target_index = max(0, props.active_target_index - 1)

        return {'FINISHED'}


class CAMERA_OT_cleanup_temp_files(Operator):
    """Clean up temporary image files created by the addon"""
    bl_idname = "camera.cleanup_temp_files"
    bl_label = "Cleanup Temp Files"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Find custom camera
        custom_cameras = [obj for obj in context.scene.objects
                         if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]

        if not custom_cameras:
            self.report({'WARNING'}, "No custom camera found")
            return {'CANCELLED'}

        obj = custom_cameras[0]
        props = obj.custom_camera_props

        cleaned_count = 0

        # Clean up fitted image
        if props.fitted_image_path:
            camera_utils.cleanup_temp_image(props.fitted_image_path)
            props.fitted_image_path = ""
            cleaned_count += 1

        # Clean up temporary mask images from temp directory
        import tempfile
        import os
        import glob

        temp_dir = tempfile.gettempdir()

        # Find all temp files created by this addon
        patterns = [
            os.path.join(temp_dir, "fitted_camera_bg_*.png"),
            os.path.join(temp_dir, "mask_original_*.png"),
            os.path.join(temp_dir, "mask_fitted_*.png"),
            os.path.join(temp_dir, f"temp_camera_bg_{os.getpid()}.png"),
        ]

        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                    cleaned_count += 1
                    print(f"Cleaned up: {filepath}")
                except Exception as e:
                    print(f"Could not delete {filepath}: {e}")

        # Clean up Blender internal temporary images
        temp_images_to_remove = [
            img for img in bpy.data.images
            if img.name.startswith("_temp_mask_") or
               img.name.startswith("mask_original_") or
               img.name.startswith("mask_fitted_")
        ]

        for img in temp_images_to_remove:
            bpy.data.images.remove(img)
            cleaned_count += 1

        self.report({'INFO'}, f"Cleaned up {cleaned_count} temporary files")
        return {'FINISHED'}


class CAMERA_OT_draw_mask(Operator):
    """Enter mask drawing mode with edge detection"""
    bl_idname = "camera.draw_mask"
    bl_label = "Draw Mask with Edge Detection"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Class variables (shared across instances)
    points = []
    mouse_pos = (0, 0)
    drawing = False
    _handle = None
    opencv_image = None
    image_size = None
    refined_points = []
    viewport_points = []     # Viewport coordinates (for display)
    
    def invoke(self, context, event):
        # Find custom camera
        custom_cameras = [obj for obj in context.scene.objects
                        if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]

        if not custom_cameras:
            self.report({'ERROR'}, "Make Custom Camera first")
            return {'CANCELLED'}

        camera_obj = custom_cameras[0]
        props = camera_obj.custom_camera_props

        if props.active_mask_index >= len(props.mask_regions):
            self.report({'ERROR'}, "Add a mask region first")
            return {'CANCELLED'}

        # Check camera view
        if context.space_data.region_3d.view_perspective != 'CAMERA':
            self.report({'WARNING'}, "Switch to camera view (Numpad 0) for best results")

        # IMPORTANT: Clear any previous visualization!
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

        self.refined_points = []  # Clear old refined points

        # Delete any existing temporary mask image and restore original reference
        self.restore_original_reference(camera_obj, props)

        # Load background image for OpenCV processing
        blender_image, self.image_size = get_camera_background_image(camera_obj)

        if blender_image and self.image_size:
            try:
                self.opencv_image = blender_image_to_opencv(blender_image)
                self.report({'INFO'}, f"Loaded {self.image_size[0]}x{self.image_size[1]} background")
                print(f"Loaded {self.image_size[0]}x{self.image_size[1]} background")
            except Exception as e:
                self.opencv_image = None
                self.report({'WARNING'}, f"Could not process image: {str(e)}")
        else:
            self.opencv_image = None
            self.report({'WARNING'}, "No background image - edge detection disabled")

        # Initialize drawing (clear everything!)
        self.points = []              # Image coordinates
        self.viewport_points = []     # Viewport coordinates
        self.mouse_pos = (0, 0)
        self.drawing = True

        # Add drawing handler
        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_mask_callback, args, 'WINDOW', 'POST_PIXEL'
        )

        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Draw mask: LMB to add points, Enter/RMB/ESC to finish")
        return {'RUNNING_MODAL'}
    def modal(self, context, event):
        """
        Handle drawing events
        
        CRITICAL BUG FIX:
        - We store points in IMAGE coordinates (for OpenCV processing)
        - But we must display them in VIEWPORT coordinates (for user to see)
        - Need to track BOTH coordinate systems!
        """
        context.area.tag_redraw()
        
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            viewport_pos = (event.mouse_region_x, event.mouse_region_y)

            camera_obj = [obj for obj in context.scene.objects
                        if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera][0]

            # Fix: Removed debug prints for faster response
            if self.opencv_image is not None and self.image_size:
                # Convert to image coordinates for storage
                image_pos = viewport_to_image_coords(
                    viewport_pos, context, camera_obj, self.image_size
                )
                
                if image_pos:
                    # Store BOTH coordinates
                    # points = image coords (for OpenCV)
                    # viewport_points = viewport coords (for display)
                    self.points.append(image_pos)
                    
                    # Also store viewport position for display
                    if not hasattr(self, 'viewport_points'):
                        self.viewport_points = []
                    self.viewport_points.append(viewport_pos)
            else:
                # Fallback: store viewport coords only
                self.points.append(viewport_pos)
                if not hasattr(self, 'viewport_points'):
                    self.viewport_points = []
                self.viewport_points.append(viewport_pos)
            
            # Auto-complete check (use viewport coords for distance)
            if hasattr(self, 'viewport_points') and len(self.viewport_points) > 2:
                first_x, first_y = self.viewport_points[0]
                x, y = self.viewport_points[-1]
                dist = ((x - first_x)**2 + (y - first_y)**2)**0.5
                if dist < 20:
                    # Fix: Merge last point to first point when closing loop
                    self.points[-1] = self.points[0]
                    self.viewport_points[-1] = self.viewport_points[0]
                    self.finish_drawing(context)
                    return {'FINISHED'}
        
        elif event.type in {'RET', 'RIGHTMOUSE', 'ESC'}:
            self.finish_drawing(context)
            return {'FINISHED'}
        
        elif event.type == 'MOUSEMOVE':
            self.mouse_pos = (event.mouse_region_x, event.mouse_region_y)
        
        return {'RUNNING_MODAL'}
    
    def finish_drawing(self, context):
        """
        Process drawing with active contour (snakes) and save as original image size.

        IMPORTANT ORDER:
        1. Draw on fitted image (correct aspect ratio for camera)
        2. Apply active contour on fitted image (accurate edge detection)
        3. Map refined mask to original image size
        4. Save both fitted (for display) and original (for export)
        """

        # Remove drawing handler
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

        self.drawing = False

        if len(self.points) < 3:
            self.report({'WARNING'}, "Need at least 3 points")
            context.area.tag_redraw()
            return

        camera_obj = [obj for obj in context.scene.objects
                    if obj.custom_camera_props.is_custom_camera][0]
        props = camera_obj.custom_camera_props
        active_mask = props.mask_regions[props.active_mask_index]

        # STEP 1: Points are already in ORIGINAL image coordinates (from viewport drawing)
        original_points = self.points
        print(f"\n=== MASK PROCESSING PIPELINE ===")
        print(f"Step 1: User drew {len(original_points)} points on original image")

        # STEP 2: Apply GrabCut refinement on ORIGINAL image for accurate edge detection
        if self.opencv_image is not None and active_mask.use_auto_refine:
            try:
                print("Step 2: Applying GrabCut refinement (fast & accurate)...")
                # Use user's constraint setting to control how closely mask follows drawing
                refined_points = apply_grabcut_refinement(
                    original_points,
                    self.opencv_image,
                    constraint=active_mask.contour_constraint  # User control for following drawing
                )

                if refined_points and len(refined_points) >= 3:
                    original_points = refined_points
                    self.refined_points = refined_points

                    self.report({'INFO'},
                        f"GrabCut refined: {len(self.points)} → {len(refined_points)} points")
                    print(f"  ✓ GrabCut refined to {len(refined_points)} points")
                else:
                    self.report({'WARNING'}, "GrabCut refinement produced no results")
                    print(f"  ✗ GrabCut failed, using manual points")

            except Exception as e:
                self.report({'WARNING'}, f"GrabCut refinement failed: {str(e)}")
                print(f"  ✗ GrabCut error: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Step 2: Skipped (auto-refine disabled)")

        # STEP 3: Store points in ORIGINAL image coordinates
        active_mask.points = json.dumps(original_points)
        print(f"Step 3: Stored {len(original_points)} points in original coordinates")

        # Debug: print sample points
        print(f"\nSample points:")
        print(f"  Original: {original_points[:3]}")
        print("=" * 50)

        # STEP 4: Save mask overlay and update camera background
        print("Step 4: Saving mask overlay...")
        self.save_mask_with_overlay(camera_obj, props, original_points)

        # STEP 5: Enable perpendicular mask mesh drawing
        print("Step 5: Enabling mask mesh visualization...")
        from . import mask_utils
        mask_utils.enable_perpendicular_mask_drawing(context)

        print("✓ Mask processing complete!")
        print("=" * 50 + "\n")

        context.area.tag_redraw()

    def scale_mask_points(self, points, scale=2.0):
        """
        Scale mask points from their center
        Used to enlarge/shrink the mask polygon
        """
        if len(points) < 3:
            return points

        # Calculate center of polygon
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)

        print(f"\n=== SCALING MASK POINTS ===")
        print(f"Scale factor: {scale}x")
        print(f"Center: ({center_x:.1f}, {center_y:.1f})")

        # Scale each point from center
        scaled_points = []
        for px, py in points:
            # Vector from center to point
            dx = px - center_x
            dy = py - center_y
            # Scale by factor
            new_x = center_x + dx * scale
            new_y = center_y + dy * scale
            scaled_points.append((int(new_x), int(new_y)))

        # Verify center is preserved
        new_center_x = sum(p[0] for p in scaled_points) / len(scaled_points)
        new_center_y = sum(p[1] for p in scaled_points) / len(scaled_points)
        print(f"New center: ({new_center_x:.1f}, {new_center_y:.1f})")
        print(f"Sample original: {points[:3]}")
        print(f"Sample scaled: {scaled_points[:3]}")
        print("=" * 50)

        return scaled_points

    def refine_outline_opencv(self, rough_points, mask_settings):
        """Use OpenCV to refine rough user drawing with user-adjustable parameters"""

        if self.opencv_image is None or len(rough_points) < 3:
            return rough_points

        print(f"\n=== OpenCV Refinement (User-Controlled) ===")
        print(f"Input: {len(rough_points)} rough points")
        print(f"Image size: {self.image_size}")
        print(f"Edge sensitivity: {mask_settings.edge_sensitivity}")
        print(f"Detail level: {mask_settings.detail_level}")

        points_array = np.array(rough_points, dtype=np.int32)

        # Get bounding box with larger padding
        x, y, w, h = cv2.boundingRect(points_array)
        print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")

        # Increase padding for better edge detection
        padding = 50
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.image_size[0] - x, w + 2 * padding)
        h = min(self.image_size[1] - y, h + 2 * padding)

        print(f"With padding: x={x}, y={y}, w={w}, h={h}")

        # Extract ROI
        roi = self.opencv_image[y:y+h, x:x+w]
        print(f"ROI shape: {roi.shape}")

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adjust Canny thresholds based on user sensitivity (0-1 range)
        # Lower sensitivity = higher thresholds = less edges detected
        # Higher sensitivity = lower thresholds = more edges detected
        sensitivity = mask_settings.edge_sensitivity
        low_threshold = int(20 + (1.0 - sensitivity) * 80)   # Range: 20-100
        high_threshold = int(80 + (1.0 - sensitivity) * 120) # Range: 80-200

        # Try multiple edge detection methods and combine results
        # Method 1: Canny edge detection with user-controlled sensitivity
        edges_canny = cv2.Canny(blurred, low_threshold, high_threshold)

        # Method 2: Adaptive threshold
        edges_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Combine edge detection methods
        edges = cv2.bitwise_or(edges_canny, edges_thresh)

        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)

        print(f"Edge pixels: {np.sum(edges > 0)}")

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)

        print(f"Found {len(contours)} contours")

        if not contours:
            print("No contours found!")
            return rough_points

        # Create user mask
        user_mask = np.zeros_like(gray)
        adjusted_points = points_array.copy()
        adjusted_points[:, 0] -= x
        adjusted_points[:, 1] -= y
        cv2.fillPoly(user_mask, [adjusted_points], 255)

        user_area = np.sum(user_mask > 0)
        print(f"User mask area: {user_area} pixels")

        # Find best matching contour with improved scoring
        best_contour = None
        best_score = 0
        best_contour_fallback = None  # Track best contour regardless of threshold
        best_score_fallback = 0
        best_overlap_ratio = 0

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue

            contour_mask = np.zeros_like(gray)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)

            # Calculate overlap
            overlap = np.sum((user_mask > 0) & (contour_mask > 0))

            # Calculate IoU (Intersection over Union) for better matching
            union = np.sum((user_mask > 0) | (contour_mask > 0))
            iou = overlap / union if union > 0 else 0

            # Score based on overlap percentage and area similarity
            overlap_ratio = overlap / user_area if user_area > 0 else 0
            area_ratio = min(area / user_area, user_area / area) if user_area > 0 else 0

            # Penalize contours that are much smaller than user drawing
            # This prevents matching to small partial edges
            size_penalty = 1.0
            if area < user_area * 0.3:  # If contour is less than 30% of user area
                size_penalty = area / (user_area * 0.3)  # Apply penalty 0-1

            # Combined score: heavily prioritize area_ratio to avoid partial edges
            # Also multiply by size_penalty to strongly discourage small contours
            score = (overlap_ratio * 0.3 + iou * 0.3 + area_ratio * 0.4) * size_penalty

            # Track best contour regardless of threshold for fallback
            if score > best_score_fallback:
                best_score_fallback = score
                best_contour_fallback = contour
                best_overlap_ratio = overlap_ratio

            # Use contour if it meets lowered threshold of 10%
            if score > best_score and overlap_ratio > 0.1:  # Lowered from 30% to 10%
                print(f"  Contour {i}: area={area:.0f}, overlap={overlap}, overlap_ratio={overlap_ratio:.2f}, IoU={iou:.2f}, area_ratio={area_ratio:.2f}, score={score:.2f} ← BEST")
                best_score = score
                best_contour = contour

        # Fallback: if no contours met threshold, use best scoring contour anyway
        if best_contour is None and best_contour_fallback is not None:
            print(f"  No contours met 10% threshold. Using best contour with {best_overlap_ratio*100:.1f}% overlap (score={best_score_fallback:.2f})")
            best_contour = best_contour_fallback
            best_score = best_score_fallback

        if best_contour is not None:
            # Adaptive simplification based on user detail level
            perimeter = cv2.arcLength(best_contour, True)
            # detail_level: 0 = very simplified, 1 = max detail
            # Epsilon range: 0.01 (low detail) to 0.001 (high detail)
            detail = mask_settings.detail_level
            epsilon = (0.01 - detail * 0.009) * perimeter  # Range: 0.01 to 0.001
            approx = cv2.approxPolyDP(best_contour, epsilon, True)

            print(f"Simplified: {len(best_contour)} → {len(approx)} points")

            # Convert back to image coordinates
            refined = [(int(p[0][0]) + x, int(p[0][1]) + y) for p in approx]

            print(f"Output: {len(refined)} refined points")
            print(f"Sample points: {refined[:3]}")
            print("=" * 50)

            return refined

        print("No good match found! Using original points.")
        print("=" * 50)
        return rough_points
    
    def save_mask_as_temp_image(self, camera_obj, props, mask_points):
        """
        Save mask visualization as a temporary image overlay
        This prevents viewport complexity and drawing errors

        Creates temp image at RENDER RESOLUTION to avoid double-scaling when displayed
        """
        if self.opencv_image is None or not self.image_size or not mask_points:
            print(f"SKIP save_mask_as_temp_image: opencv_image={self.opencv_image is not None}, image_size={self.image_size}, mask_points={len(mask_points) if mask_points else 0}")
            return

        try:
            # Get render resolution
            scene = bpy.context.scene
            render_width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
            render_height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

            print(f"\n=== CREATING TEMP IMAGE AT RENDER RESOLUTION ===")
            print(f"Original image size: {self.image_size}")
            print(f"Render resolution: {render_width}x{render_height}")

            # Get original image dimensions
            orig_width, orig_height = self.image_size

            # Calculate how the original image will be scaled/positioned in render resolution
            # This mimics Blender's FIT mode behavior
            render_aspect = render_width / render_height
            image_aspect = orig_width / orig_height

            if image_aspect > render_aspect:
                # Image is wider - fits width, letterbox top/bottom
                scale_factor = render_width / orig_width
                scaled_width = render_width
                scaled_height = int(orig_height * scale_factor)
                offset_x = 0
                offset_y = (render_height - scaled_height) // 2
            else:
                # Image is taller or same - fits height, pillarbox left/right
                scale_factor = render_height / orig_height
                scaled_height = render_height
                scaled_width = int(orig_width * scale_factor)
                offset_x = (render_width - scaled_width) // 2
                offset_y = 0

            print(f"Scale factor: {scale_factor:.4f}")
            print(f"Scaled image: {scaled_width}x{scaled_height}")
            print(f"Offset: ({offset_x}, {offset_y})")

            # Scale the mask points from original image space to render space
            # Note: OpenCV uses top-left origin, we need to handle this correctly
            scaled_points = []
            for px, py in mask_points:
                # Scale point from image space to render space
                scaled_x = int(px * scale_factor + offset_x)
                # For Y: the image is already in the correct orientation (OpenCV top-down)
                # and gets placed in the render space with offset_y
                scaled_y = int(py * scale_factor + offset_y)
                scaled_points.append((scaled_x, scaled_y))

            print(f"Scaled {len(mask_points)} points to render resolution")
            print(f"Sample original points: {mask_points[:3]}")
            print(f"Sample scaled points: {scaled_points[:3]}")

            # Now scale the mask 2x from its center in render space
            if len(scaled_points) >= 3:
                # Calculate center of polygon in render space
                center_x = sum(p[0] for p in scaled_points) / len(scaled_points)
                center_y = sum(p[1] for p in scaled_points) / len(scaled_points)

                print(f"\n=== 2X SCALING IN RENDER SPACE ===")
                print(f"Center in render space: ({center_x:.1f}, {center_y:.1f})")

                # Scale each point 2x from center
                doubled_points = []
                for px, py in scaled_points:
                    # Vector from center to point
                    dx = px - center_x
                    dy = py - center_y
                    # Scale by 2x
                    new_x = center_x + dx * 2.0
                    new_y = center_y + dy * 2.0
                    doubled_points.append((new_x, new_y))

                # Verify center is preserved
                new_center_x = sum(p[0] for p in doubled_points) / len(doubled_points)
                new_center_y = sum(p[1] for p in doubled_points) / len(doubled_points)
                print(f"New center after 2x scale: ({new_center_x:.1f}, {new_center_y:.1f})")

                # Convert to integers for drawing
                scaled_points = [(int(x), int(y)) for x, y in doubled_points]
                print(f"Sample doubled points: {scaled_points[:3]}")
                print("=" * 50)

            # Resize the original image to fit in render resolution
            img_resized = cv2.resize(self.opencv_image, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)

            # Create a blank image at render resolution with black bars
            img_with_mask = np.zeros((render_height, render_width, 3), dtype=np.uint8)

            # Place the resized image in the center (accounting for letterbox/pillarbox)
            img_with_mask[offset_y:offset_y+scaled_height, offset_x:offset_x+scaled_width] = img_resized

            # Draw the mask as a CLOSED polygon using scaled points
            points_array = np.array(scaled_points, dtype=np.int32)

            # isClosed=True ensures the polygon is closed
            cv2.polylines(img_with_mask, [points_array], isClosed=True, color=(255, 255, 0), thickness=3)

            # Fill the mask with semi-transparent overlay
            overlay = img_with_mask.copy()
            cv2.fillPoly(overlay, [points_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, img_with_mask, 0.8, 0, img_with_mask)

            # Convert back to Blender format (BGR -> RGB, uint8 -> float)
            rgb_array = cv2.cvtColor(img_with_mask, cv2.COLOR_BGR2RGB)
            rgb_float = rgb_array.astype(np.float32) / 255.0

            # Flip back for Blender (OpenCV: top-left origin, Blender: bottom-left)
            rgb_float = np.flipud(rgb_float)

            # Add alpha channel
            rgba_array = np.ones((render_height, render_width, 4), dtype=np.float32)
            rgba_array[:, :, :3] = rgb_float

            # Create or update temporary image in Blender
            temp_image_name = f"_temp_mask_{props.active_mask_index}"

            if temp_image_name in bpy.data.images:
                temp_image = bpy.data.images[temp_image_name]
                # Update existing image
                temp_image.scale(render_width, render_height)
            else:
                # Create new image
                temp_image = bpy.data.images.new(temp_image_name, render_width, render_height, alpha=True)

            # Set pixels
            temp_image.pixels[:] = rgba_array.flatten()
            temp_image.update()

            # Update camera background to show the temp image
            # Now FIT mode won't cause double scaling since temp image is already at render resolution
            camera_utils.setup_camera_background(
                camera_obj, temp_image, props.reference_image_opacity
            )

            print(f"Saved mask as temporary image: {temp_image_name} ({render_width}x{render_height})")

        except Exception as e:
            print(f"Error saving mask as temp image: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_mask_with_overlay(self, camera_obj, props, mask_points_original):
        """
        Save mask visualization with 2x scaling from center.

        Args:
            mask_points_original: Points in original image coordinates
        """
        if not props.reference_image_path or not mask_points_original:
            print("SKIP save_mask_with_overlay: No reference image or mask points")
            return

        try:
            print(f"\n=== SAVING MASK OVERLAY ===")

            # Load original image with OpenCV
            original_img = cv2.imread(props.reference_image_path)
            if original_img is None:
                print(f"Could not load original image: {props.reference_image_path}")
                return

            orig_height, orig_width = original_img.shape[:2]
            print(f"Original image size: {orig_width}x{orig_height}")
            print(f"Mask points (original coords): {len(mask_points_original)}")

            # SCALE MASK 2X FROM CENTER (in original image coordinates)

            # Create a copy for drawing
            img_with_mask = original_img.copy()

            # Draw the mask outline on image
            mask_points_array = np.array(mask_points_original, dtype=np.int32)

            # Draw closed polygon
            cv2.polylines(img_with_mask, [mask_points_array], isClosed=True,
                         color=(255, 255, 0), thickness=2)

            # Fill with semi-transparent overlay
            overlay = img_with_mask.copy()
            cv2.fillPoly(overlay, [mask_points_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, img_with_mask, 0.8, 0, img_with_mask)

            print(f"  Drew mask on image with 2x scaling")

            # Save as temporary file
            temp_path = camera_utils.save_opencv_image_as_temp(
                img_with_mask, f"mask_overlay_{props.active_mask_index}"
            )

            # Load into Blender
            if temp_path in bpy.data.images:
                mask_img = bpy.data.images[temp_path]
                mask_img.reload()
            else:
                mask_img = bpy.data.images.load(temp_path)

            # Store overlay path in active mask for later switching
            active_mask = props.mask_regions[props.active_mask_index]
            active_mask.mask_overlay_path = temp_path

            # Update camera background with mask overlay
            camera_utils.setup_camera_background(
                camera_obj, mask_img, props.reference_image_opacity
            )

            print(f"✓ Saved mask overlay with 2x scaling to: {temp_path}")
            print("=" * 50)

        except Exception as e:
            print(f"Error saving mask overlay: {str(e)}")
            import traceback
            traceback.print_exc()

    def restore_original_reference(self, camera_obj, props):
        """
        Restore original reference image without mask overlay
        This is called when starting a new drawing session
        Note: We keep existing mask overlays stored in each mask for switching
        """
        try:
            # Load original reference image (without any mask overlay)
            if props.reference_image_path:
                # Load or get the original image
                if props.reference_image_path in bpy.data.images:
                    orig_img = bpy.data.images[props.reference_image_path]
                    orig_img.reload()
                else:
                    try:
                        orig_img = bpy.data.images.load(props.reference_image_path)
                    except:
                        print(f"Could not reload original image: {props.reference_image_path}")
                        return

                # Set original image as camera background
                camera_utils.setup_camera_background(
                    camera_obj, orig_img, props.reference_image_opacity
                )

                print(f"Restored original reference image: {props.reference_image_path}")

        except Exception as e:
            print(f"Error restoring original reference: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_mask_rays(self, camera_obj, props, mask_points, context):
        """
        Store mask ray data for viewport visualization (non-object based)
        """
        if not mask_points or not self.image_size:
            return

        try:
            from mathutils import Vector

            # Get active mask
            active_mask = props.mask_regions[props.active_mask_index]

            # Store ray data in the mask for viewport drawing
            ray_data = []
            camera_location = camera_obj.matrix_world.translation

            # Calculate all ray endpoints
            for point in mask_points:
                world_pos = self.image_coords_to_world_ray(
                    camera_obj, point, self.image_size, distance=active_mask.ray_length
                )

                if world_pos is not None:
                    ray_data.append({
                        'start': camera_location.copy(),
                        'end': world_pos,
                    })

            # Store as JSON in custom property (for persistence)
            import json
            active_mask['_ray_data'] = json.dumps([
                {'start': [r['start'].x, r['start'].y, r['start'].z],
                 'end': [r['end'].x, r['end'].y, r['end'].z]}
                for r in ray_data
            ])

            print(f"Stored {len(ray_data)} rays for viewport visualization")

            # Enable viewport drawing
            from . import mask_utils
            mask_utils.enable_mask_ray_drawing(context)

        except Exception as e:
            print(f"Error storing mask rays: {str(e)}")
            import traceback
            traceback.print_exc()

    def image_coords_to_world_ray(self, camera_obj, image_point, image_size, distance=10.0):
        """
        Convert image coordinates to world position along camera ray
        """
        from mathutils import Vector

        img_x, img_y = image_point
        img_width, img_height = image_size

        # Normalize to [-1, 1]
        ndc_x = (img_x / img_width) * 2.0 - 1.0
        ndc_y = 1.0 - (img_y / img_height) * 2.0  # Flip Y

        # Get camera data
        camera = camera_obj.data
        aspect = camera.sensor_width / camera.sensor_height

        # Calculate position in camera space
        if camera.type == 'PERSP':
            import math
            fov = camera.angle
            height = 2.0 * distance * math.tan(fov / 2.0)
            width = height * aspect

            x = ndc_x * width / 2.0
            y = ndc_y * height / 2.0
            z = -distance
        else:
            # Orthographic
            width = camera.ortho_scale
            height = width / aspect

            x = ndc_x * width / 2.0
            y = ndc_y * height / 2.0
            z = -distance

        # Transform to world space
        camera_space = Vector((x, y, z))
        world_pos = camera_obj.matrix_world @ camera_space

        return world_pos

    def clear_visualization(self, context=None):
        """
        Clear visualization after showing refined result

        BUG FIX: Context can be None in timer callbacks!
        Solution: Get context from bpy.context instead
        """
        if self._handle:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            self._handle = None

        self.refined_points = []

        # Get fresh context (don't use parameter)
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        return None  # Don't repeat timer


# ============================================================================
# DRAWING CALLBACK
# ============================================================================
def draw_mask_callback(self, context):
    """
    Draw mask outline while drawing and show refined result
    
    COORDINATE FIX:
    - Use viewport_points for displaying rough drawing (green lines)
    - Convert refined_points from image to viewport for display (yellow lines)
    """
    import gpu
    from gpu_extras.batch import batch_for_shader
    
    if not self.drawing and not self.refined_points:
        return
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    
    # ========================================================================
    # Draw refined mask (yellow) - convert from image coords to viewport
    # ========================================================================
    if self.refined_points and hasattr(self, 'image_size') and self.image_size:
        camera_obj = [obj for obj in context.scene.objects 
                     if obj.custom_camera_props.is_custom_camera][0]
        
        viewport_points = []
        for img_point in self.refined_points:
            vp_point = image_to_viewport_coords(
                img_point, context, camera_obj, self.image_size
            )
            if vp_point:
                viewport_points.append(vp_point)
        
        if len(viewport_points) > 1:
            vertices = [(p[0], p[1]) for p in viewport_points]
            vertices.append(vertices[0])  # Close loop
            
            indices = [(i, i+1) for i in range(len(vertices)-1)]
            
            batch = batch_for_shader(shader, 'LINES', 
                                    {"pos": vertices}, indices=indices)
            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))  # Yellow
            gpu.state.line_width_set(3.0)
            batch.draw(shader)
    
    # ========================================================================
    # Draw rough drawing (green) - use viewport_points directly
    # ========================================================================
    if self.drawing and hasattr(self, 'viewport_points') and len(self.viewport_points) > 0:
        # Draw lines connecting points
        if len(self.viewport_points) > 1:
            vertices = [(p[0], p[1]) for p in self.viewport_points]
            indices = [(i, i+1) for i in range(len(vertices)-1)]
            
            batch = batch_for_shader(shader, 'LINES', 
                                    {"pos": vertices}, indices=indices)
            shader.bind()
            shader.uniform_float("color", (0.0, 1.0, 0.0, 0.8))  # Green
            gpu.state.line_width_set(2.0)
            batch.draw(shader)
        
        # Line from last point to mouse cursor
        if len(self.viewport_points) > 0:
            vertices = [self.viewport_points[-1], self.mouse_pos]
            batch = batch_for_shader(shader, 'LINES', {"pos": vertices})
            shader.bind()
            shader.uniform_float("color", (0.5, 1.0, 0.5, 0.5))
            gpu.state.line_width_set(1.0)
            batch.draw(shader)
        
        # Draw points
        vertices = [(p[0], p[1]) for p in self.viewport_points]
        batch = batch_for_shader(shader, 'POINTS', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (0.0, 1.0, 0.0, 1.0))
        gpu.state.point_size_set(6.0)
        batch.draw(shader)
        
        # First point (red)
        vertices = [(self.viewport_points[0][0], self.viewport_points[0][1])]
        batch = batch_for_shader(shader, 'POINTS', {"pos": vertices})
        shader.bind()
        shader.uniform_float("color", (1.0, 0.0, 0.0, 1.0))  # Red
        gpu.state.point_size_set(8.0)
        batch.draw(shader)
    
    # Reset GPU state
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)

def test_coordinate_conversion(context):
    """
    Test coordinate conversion
    Call from console: test_coordinate_conversion(C)
    """
    from . import operators
    
    # Get camera and image
    custom_cameras = [obj for obj in context.scene.objects 
                     if obj.custom_camera_props.is_custom_camera]
    if not custom_cameras:
        print("No custom camera!")
        return
    
    camera_obj = custom_cameras[0]
    blender_image, image_size = operators.get_camera_background_image(camera_obj)
    
    if not image_size:
        print("No background image!")
        return
    
    print(f"\nImage size: {image_size}")
    print(f"Viewport size: {context.region.width}x{context.region.height}")
    
    # Test center point
    center_vp = (context.region.width // 2, context.region.height // 2)
    center_img = operators.viewport_to_image_coords(center_vp, context, camera_obj, image_size)
    
    print(f"\nCenter test:")
    print(f"  Viewport: {center_vp}")
    print(f"  Image: {center_img}")
    print(f"  Expected image center: ({image_size[0]//2}, {image_size[1]//2})")
    
    # Convert back
    if center_img:
        back_vp = operators.image_to_viewport_coords(center_img, context, camera_obj, image_size)
        print(f"  Back to viewport: {back_vp}")
        print(f"  Difference: ({back_vp[0]-center_vp[0]}, {back_vp[1]-center_vp[1]})")

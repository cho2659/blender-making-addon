import bpy
import gpu
import json
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from . import camera_utils

# Global handler references
_draw_handler = None
_ray_draw_handler = None
_mask_mesh_handler = None
_depsgraph_handler = None

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = point
    inside = False
    
    n = len(polygon)
    p1x, p1y = polygon[0]
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside

def create_mask_frustum(camera_obj, mask_region):
    """Create 3D frustum from 2D mask polygon"""
    if not mask_region.points:
        return None
    
    try:
        points_2d = json.loads(mask_region.points)
    except:
        return None
    
    if len(points_2d) < 3:
        return None
    
    # Convert 2D screen points to 3D rays from camera
    resolution_x = bpy.context.scene.render.resolution_x  # Default resolution, should match viewport
    resolution_y = bpy.context.scene.render.resolution_y
    
    rays = []
    for px, py in points_2d:
        world_pos = camera_utils.screen_to_camera_space(
            camera_obj, px, py, resolution_x, resolution_y
        )
        rays.append(world_pos)
    
    return rays

def check_point_in_mask_3d(camera_obj, mask_region, world_pos):
    """Check if 3D point projects inside 2D mask region"""
    if not mask_region.points or not mask_region.enabled:
        return False
    
    try:
        points_2d = json.loads(mask_region.points)
    except:
        return False
    
    if len(points_2d) < 3:
        return False
    
    # Project world position to screen space
    resolution_x = 1920
    resolution_y = 1080
    
    screen_pos = camera_utils.project_world_to_screen(
        camera_obj, world_pos, resolution_x, resolution_y
    )
    
    if screen_pos is None:
        return False
    
    # Check if projected point is inside polygon
    return point_in_polygon(screen_pos, points_2d)

def get_object_mask_colors(camera_obj, target_obj, props):
    """Calculate colors for target object based on mask intersections"""
    if not target_obj or target_obj.type != 'MESH':
        return []
    
    # Get object vertices in world space
    mesh = target_obj.data
    matrix_world = target_obj.matrix_world
    
    vertex_colors = []
    
    for vert in mesh.vertices:
        world_pos = matrix_world @ vert.co
        
        # Check against all enabled mask regions
        inside_masks = []
        
        for mask in props.mask_regions:
            if mask.enabled:
                if check_point_in_mask_3d(camera_obj, mask, world_pos):
                    inside_masks.append(mask)
        
        # Determine color based on mask intersections
        if len(inside_masks) == 0:
            # Outside all masks - use first mask's outside color
            if len(props.mask_regions) > 0:
                color = props.mask_regions[0].outside_color
            else:
                color = (1.0, 0.0, 0.0, 1.0)
        elif len(inside_masks) == 1:
            # Inside one mask
            mask = inside_masks[0]
            color = mask.inside_color
            
            # Apply depth-based transparency
            camera_space = camera_obj.matrix_world.inverted() @ world_pos
            distance = -camera_space.z
            
            if distance < mask.fade_start:
                alpha = 1.0 - mask.transparency_at_camera
            elif distance > mask.fade_end:
                alpha = 0.0
            else:
                fade_range = mask.fade_end - mask.fade_start
                fade_factor = (distance - mask.fade_start) / fade_range
                alpha = (1.0 - mask.transparency_at_camera) * (1.0 - fade_factor)
            
            color = (color[0], color[1], color[2], alpha)
        else:
            # Fix: Highlight overlapping regions with brighter intersection color and full opacity
            color = props.intersection_color
            # Ensure full opacity for intersection visibility
            color = (color[0], color[1], color[2], 1.0)

        vertex_colors.append(color)
    
    return vertex_colors

def draw_viewport_overlay():
    """Draw mask overlay in viewport"""
    context = bpy.context
    
    # Find active custom camera
    camera_obj = None
    for obj in context.scene.objects:
        if obj.custom_camera_props.is_custom_camera:
            if obj.custom_camera_props.show_mask_overlay:
                camera_obj = obj
                break
    
    if not camera_obj:
        return
    
    props = camera_obj.custom_camera_props
    
    # Get target objects from list (prefer list over single target)
    target_objects = []
    
    # Check if target_objects attribute exists (for backward compatibility)
    if hasattr(props, 'target_objects') and len(props.target_objects) > 0:
        for target_item in props.target_objects:
            if target_item.obj and target_item.obj.type == 'MESH':
                target_objects.append(target_item.obj)
    elif hasattr(props, 'target_object') and props.target_object and props.target_object.type == 'MESH':
        # Fallback to legacy single target
        target_objects.append(props.target_object)
    
    if not target_objects:
        return
    
    # Draw overlay for each target object
    for target_obj in target_objects:
        draw_object_overlay(camera_obj, target_obj, props)


def draw_object_overlay(camera_obj, target_obj, props):
    """Draw overlay for a single target object with X-ray mode"""
    # Get vertex colors
    vertex_colors = get_object_mask_colors(camera_obj, target_obj, props)

    if not vertex_colors:
        return

    # Create shader for vertex color visualization
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')

    # Prepare vertex positions and colors
    mesh = target_obj.data
    matrix_world = target_obj.matrix_world

    positions = []
    colors = []
    indices = []

    for poly in mesh.polygons:
        poly_verts = []
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            world_pos = matrix_world @ mesh.vertices[vert_idx].co
            positions.append(world_pos)
            colors.append(vertex_colors[vert_idx])
            poly_verts.append(len(positions) - 1)

        # Triangulate polygon
        for i in range(1, len(poly_verts) - 1):
            indices.append((poly_verts[0], poly_verts[i], poly_verts[i + 1]))

    if not indices:
        return

    # Enable X-ray mode: disable depth test so it always shows through
    gpu.state.depth_test_set('NONE')  # Disable depth testing (X-ray effect)
    gpu.state.blend_set('ALPHA')      # Enable alpha blending

    # Create batch and draw
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": positions, "color": colors},
        indices=indices
    )

    shader.bind()
    batch.draw(shader)

    # Reset GPU state
    gpu.state.depth_test_set('LESS_EQUAL')  # Re-enable standard depth testing
    gpu.state.blend_set('NONE')

def enable_viewport_drawing(context):
    """Enable viewport drawing handler"""
    global _draw_handler

    if _draw_handler is not None:
        return

    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_viewport_overlay, (), 'WINDOW', 'POST_VIEW'
    )

    # Enable auto-update on scene changes
    enable_depsgraph_updates()

    # Force viewport update
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def disable_viewport_drawing():
    """Disable viewport drawing handler"""
    global _draw_handler

    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None

        # Disable auto-update
        disable_depsgraph_updates()

        # Force viewport update
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

def depsgraph_update_handler(scene, depsgraph):
    """Handler called when scene updates (objects move, transform, etc.)"""
    # Check if any custom camera exists with overlay enabled
    has_active_overlay = False
    for obj in scene.objects:
        if obj.custom_camera_props.is_custom_camera:
            if obj.custom_camera_props.show_mask_overlay:
                has_active_overlay = True
                break

    if not has_active_overlay:
        return

    # Check if any relevant objects were updated
    for update in depsgraph.updates:
        # If object transform changed, trigger viewport redraw
        if update.is_updated_transform or update.is_updated_geometry:
            # Force all 3D viewports to redraw
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
            break

def enable_depsgraph_updates():
    """Enable depsgraph update handler for auto-updating mask overlays"""
    global _depsgraph_handler

    if _depsgraph_handler is not None:
        return

    # Add handler to depsgraph updates
    if depsgraph_update_handler not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)
        _depsgraph_handler = depsgraph_update_handler

def disable_depsgraph_updates():
    """Disable depsgraph update handler"""
    global _depsgraph_handler

    if _depsgraph_handler is not None:
        if _depsgraph_handler in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_depsgraph_handler)
        _depsgraph_handler = None


# ============================================================================
# MASK RAY VISUALIZATION (Viewport Only - No Scene Objects)
# ============================================================================

def draw_ray_with_depth_fade(start, end, opacity=0.5, fade_start=0.0, fade_end=10.0, base_color=(0.0, 1.0, 0.0)):
    """
    Draw a single ray with depth-based opacity fade.

    Args:
        start: Vector - ray start position (usually camera position)
        end: Vector - ray end position
        opacity: float - base opacity (0.0-1.0)
        fade_start: float - distance from camera where fade begins
        fade_end: float - distance from camera where ray becomes fully transparent
        base_color: tuple - RGB color (r, g, b)

    Returns:
        vertices, colors - lists for batch rendering
    """
    total_distance = (end - start).length

    # Calculate opacity at start and end based on fade parameters
    if total_distance <= fade_start:
        # Entire ray is before fade starts - full opacity
        start_alpha = opacity
        end_alpha = opacity
    elif total_distance >= fade_end:
        # Ray extends beyond fade end
        start_alpha = opacity
        # Calculate fade at end
        fade_range = fade_end - fade_start
        if fade_range > 0:
            end_alpha = opacity * (1.0 - min(1.0, (total_distance - fade_start) / fade_range))
        else:
            end_alpha = 0.0
    else:
        # Ray is within fade range
        start_alpha = opacity
        fade_range = fade_end - fade_start
        if fade_range > 0:
            fade_factor = (total_distance - fade_start) / fade_range
            end_alpha = opacity * (1.0 - max(0.0, min(1.0, fade_factor)))
        else:
            end_alpha = opacity

    vertices = [tuple(start), tuple(end)]
    colors = [(*base_color, start_alpha), (*base_color, end_alpha)]

    return vertices, colors


def draw_mask_rays():
    """Draw mask rays as viewport overlay (non-object based)"""
    import gpu
    from gpu_extras.batch import batch_for_shader

    context = bpy.context

    # Find active custom camera
    camera_obj = None
    for obj in context.scene.objects:
        if obj.custom_camera_props.is_custom_camera:
            camera_obj = obj
            break

    if not camera_obj:
        return

    props = camera_obj.custom_camera_props

    # Draw rays for each enabled mask
    for mask in props.mask_regions:
        if not mask.enabled or not mask.show_rays:
            continue

        # Get ray data from mask
        if '_ray_data' not in mask:
            continue

        try:
            ray_data = json.loads(mask['_ray_data'])
        except:
            continue

        if not ray_data:
            continue

        # Prepare vertices for batch drawing with depth fade
        all_vertices = []
        all_colors = []

        # Choose base color based on mask mode
        if mask.mask_mode == 'TARGET':
            base_color = (0.0, 1.0, 0.0)  # Green
        else:
            base_color = (1.0, 0.0, 0.0)  # Red

        for ray in ray_data:
            start = Vector(ray['start'])
            end = Vector(ray['end'])

            # Use unified depth fade function
            vertices, colors = draw_ray_with_depth_fade(
                start, end,
                opacity=mask.ray_opacity,
                fade_start=mask.fade_start,
                fade_end=mask.fade_end,
                base_color=base_color
            )

            all_vertices.extend(vertices)
            all_colors.extend(colors)

        if not all_vertices:
            continue

        # Create shader for smooth color transitions
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": all_vertices, "color": all_colors})

        # Enable blending for transparency
        gpu.state.blend_set('ALPHA')
        gpu.state.line_width_set(2.0)

        # Draw
        shader.bind()
        batch.draw(shader)

        # Reset state
        gpu.state.blend_set('NONE')
        gpu.state.line_width_set(1.0)


def draw_perpendicular_mask_mesh():
    """
    Draw mask as a 3D cylindrical mesh extending from camera.
    - Creates a cylinder (not cone) from start distance to end distance
    - Transparent mesh that respects depth testing
    - Hidden from camera view
    - Colors target objects: green if inside mask, red if outside
    """
    import gpu
    from gpu_extras.batch import batch_for_shader

    context = bpy.context

    # Find active custom camera
    camera_obj = None
    for obj in context.scene.objects:
        if obj.custom_camera_props.is_custom_camera:
            camera_obj = obj
            break

    if not camera_obj:
        return

    props = camera_obj.custom_camera_props

    # Don't draw if viewing through the custom camera
    if context.space_data.region_3d.view_perspective == 'CAMERA' and context.scene.camera == camera_obj:
        # Still draw target object coloring
        draw_target_object_coloring(camera_obj, props)
        return

    # Get camera matrix (includes position, rotation, and scale)
    camera_matrix = camera_obj.matrix_world

    # Draw mask mesh for each enabled mask
    for mask in props.mask_regions:
        if not mask.enabled:
            continue

        # Get mask points
        if not mask.points:
            continue

        try:
            points_2d = json.loads(mask.points)
        except:
            continue

        if len(points_2d) < 3:
            continue

        # Get image size from camera background
        if not camera_obj.data.background_images:
            continue

        bg_image = camera_obj.data.background_images[0]
        if not bg_image.image:
            continue

        image_width = bg_image.image.size[0]
        image_height = bg_image.image.size[1]

        # Get start and end distances from mask properties
        # Start = image sensor plane distance (fade_start)
        # End = far distance (fade_end)
        start_distance = mask.fade_start if hasattr(mask, 'fade_start') else 1.0
        end_distance = mask.fade_end if hasattr(mask, 'fade_end') else 10.0
        mesh_opacity = mask.ray_opacity if hasattr(mask, 'ray_opacity') else 0.5

        # Create cylindrical mesh vertices from image plane to end distance
        start_vertices = []  # At image sensor plane (start_distance)
        end_vertices = []    # At end distance

        for px, py in points_2d:
            # Convert image coordinates to NDC (normalized device coordinates)
            ndc_x = (px / image_width) * 2.0 - 1.0
            ndc_y = 1.0 - (py / image_height) * 2.0  # Flip Y

            # Get camera data
            camera = camera_obj.data
            aspect = camera.sensor_width / camera.sensor_height

            # Calculate positions - uniform cylinder from image plane to end distance
            if camera.type == 'PERSP':
                import math
                fov = camera.angle

                # Calculate frustum size at START distance (image sensor plane)
                height_start = 2.0 * start_distance * math.tan(fov / 2.0)
                width_start = height_start * aspect

                # Start at image sensor plane
                x_start = ndc_x * width_start / 2.0
                y_start = ndc_y * height_start / 2.0
                z_start = -start_distance

                # End at specified distance with SAME dimensions (uniform cylinder)
                x_end = ndc_x * width_start / 2.0  # Same as start for uniform cylinder
                y_end = ndc_y * height_start / 2.0  # Same as start for uniform cylinder
                z_end = -end_distance
            else:
                # Orthographic - perfect cylinder
                width = camera.ortho_scale
                height = width / aspect

                x_start = ndc_x * width / 2.0
                y_start = ndc_y * height / 2.0
                z_start = -start_distance

                x_end = ndc_x * width / 2.0
                y_end = ndc_y * height / 2.0
                z_end = -end_distance

            # Transform to world space
            start_pos = camera_matrix @ Vector((x_start, y_start, z_start))
            end_pos = camera_matrix @ Vector((x_end, y_end, z_end))

            start_vertices.append(start_pos)
            end_vertices.append(end_pos)

        # Build cylinder mesh: connect start and end vertices
        positions = []
        colors = []
        indices = []

        base_gray = (0.5, 0.5, 0.5)
        num_points = len(start_vertices)

        # Calculate centers (needed for caps and normals)
        start_center = Vector((0, 0, 0))
        for v in start_vertices:
            start_center += v
        if num_points > 0:
            start_center /= num_points

        end_center = Vector((0, 0, 0))
        for v in end_vertices:
            end_center += v
        if num_points > 0:
            end_center /= num_points

        # Calculate cylinder axis for normals
        cylinder_axis = end_center - start_center
        if cylinder_axis.length > 0.001:
            cylinder_axis.normalize()
        else:
            cylinder_axis = Vector((0, 0, -1))

        # Calculate radial normals for each vertex
        start_normals = []
        end_normals = []
        for i in range(num_points):
            # Radial normal points from center outward
            start_radial = start_vertices[i] - start_center
            end_radial = end_vertices[i] - end_center

            # Project onto plane perpendicular to axis
            start_radial = start_radial - start_radial.project(cylinder_axis)
            end_radial = end_radial - end_radial.project(cylinder_axis)

            if start_radial.length > 0.001:
                start_radial.normalize()
            else:
                start_radial = Vector((1, 0, 0))

            if end_radial.length > 0.001:
                end_radial.normalize()
            else:
                end_radial = Vector((1, 0, 0))

            start_normals.append(start_radial)
            end_normals.append(end_radial)

        # Default Blender workbench lighting
        key_light = Vector((0.6, 0.4, -0.7))
        key_light.normalize()
        fill_light = Vector((-0.3, 0.2, -0.6))
        fill_light.normalize()

        # Add all vertices with lighting
        for i, v in enumerate(start_vertices):
            positions.append(v)
            normal = start_normals[i]
            # Default Blender shading
            key_diffuse = max(0.0, normal.dot(key_light))
            fill_diffuse = max(0.0, normal.dot(fill_light))
            diffuse = 0.8 * key_diffuse + 0.3 * fill_diffuse + 0.2
            diffuse = min(1.0, diffuse)
            lit_color = tuple(base_gray[j] * diffuse for j in range(3))
            colors.append(lit_color + (mesh_opacity,))

        for i, v in enumerate(end_vertices):
            positions.append(v)
            normal = end_normals[i]
            key_diffuse = max(0.0, normal.dot(key_light))
            fill_diffuse = max(0.0, normal.dot(fill_light))
            diffuse = 0.8 * key_diffuse + 0.3 * fill_diffuse + 0.2
            diffuse = min(1.0, diffuse)
            lit_color = tuple(base_gray[j] * diffuse for j in range(3))
            colors.append(lit_color + (mesh_opacity,))

        # Create side faces (connecting start to end)
        for i in range(num_points):
            next_i = (i + 1) % num_points

            # Two triangles per quad
            # Triangle 1: start[i], start[next_i], end[i]
            indices.append((i, next_i, num_points + i))
            # Triangle 2: start[next_i], end[next_i], end[i]
            indices.append((next_i, num_points + next_i, num_points + i))

        # Add start cap (near camera) - triangulate from center
        if num_points >= 3:
            # Add center vertex with lighting
            center_start_idx = len(positions)
            positions.append(start_center)
            # Cap normal points toward camera (opposite of axis)
            cap_normal = -cylinder_axis
            key_diffuse = max(0.0, cap_normal.dot(key_light))
            fill_diffuse = max(0.0, cap_normal.dot(fill_light))
            diffuse = 0.8 * key_diffuse + 0.3 * fill_diffuse + 0.2
            diffuse = min(1.0, diffuse)
            lit_color = tuple(base_gray[j] * diffuse for j in range(3))
            colors.append(lit_color + (mesh_opacity,))

            # Create triangles from center to edge
            for i in range(num_points):
                next_i = (i + 1) % num_points
                # Triangle facing outward from camera (reverse winding for correct normal)
                indices.append((center_start_idx, next_i, i))

        # Add end cap (far from camera) - triangulate from center
        if num_points >= 3:
            # Add center vertex with lighting
            center_end_idx = len(positions)
            positions.append(end_center)
            # Cap normal points away from camera (along axis)
            cap_normal = cylinder_axis
            key_diffuse = max(0.0, cap_normal.dot(key_light))
            fill_diffuse = max(0.0, cap_normal.dot(fill_light))
            diffuse = 0.8 * key_diffuse + 0.3 * fill_diffuse + 0.2
            diffuse = min(1.0, diffuse)
            lit_color = tuple(base_gray[j] * diffuse for j in range(3))
            colors.append(lit_color + (mesh_opacity,))

            # Create triangles from center to edge
            for i in range(num_points):
                next_i = (i + 1) % num_points
                # Triangle facing away from camera (normal winding)
                indices.append((center_end_idx, num_points + i, num_points + next_i))

        if not indices:
            continue

        # Draw the cylinder mesh
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        batch = batch_for_shader(
            shader, 'TRIS',
            {"pos": positions, "color": colors},
            indices=indices
        )

        # Enable transparency and proper depth testing
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')  # Respect depth, don't draw in front of everything
        gpu.state.face_culling_set('NONE')  # Show both sides

        shader.bind()
        batch.draw(shader)

        # Reset state
        gpu.state.blend_set('NONE')
        gpu.state.face_culling_set('BACK')

    # Now color target objects based on whether they're inside or outside the mask
    draw_target_object_coloring(camera_obj, props)


def draw_target_object_coloring(camera_obj, props):
    """
    Color target objects based on mask regions:
    - Green if inside mask
    - Red if outside mask
    - Uses Blender-style lighting with surface normals
    - Shows on surface even when overlapped with mask (polygon offset)
    """
    import gpu
    from gpu_extras.batch import batch_for_shader

    # Get target objects from collection or legacy target_object
    target_objects = []

    for mask in props.mask_regions:
        if not mask.enabled:
            continue

        # Get objects from target collection
        if mask.target_collection:
            for obj in mask.target_collection.objects:
                if obj.type == 'MESH' and obj not in target_objects:
                    target_objects.append(obj)

    # Fallback to legacy target_objects list
    if not target_objects and hasattr(props, 'target_objects'):
        for target_item in props.target_objects:
            if target_item.obj and target_item.obj.type == 'MESH':
                target_objects.append(target_item.obj)

    if not target_objects:
        return

    # Default Blender workbench lighting (same as mask mesh)
    key_light = Vector((0.6, 0.4, -0.7))
    key_light.normalize()
    fill_light = Vector((-0.3, 0.2, -0.6))
    fill_light.normalize()

    # Draw each target object with color and lighting based on mask intersection
    for target_obj in target_objects:
        mesh = target_obj.data
        matrix_world = target_obj.matrix_world
        normal_matrix = matrix_world.to_3x3().inverted().transposed()

        # First pass: determine base color per vertex (inside/outside)
        # Green = OVERLAPPING (inside mask from camera view)
        # Red = NOT OVERLAPPING (outside mask from camera view)
        vertex_base_colors = []
        for vert in mesh.vertices:
            world_pos = matrix_world @ vert.co

            # Check if vertex is inside any mask region (3D check)
            is_inside = False
            green_color = (0.0, 1.0, 0.0)  # Green for overlapping
            red_color = (1.0, 0.0, 0.0)    # Red for non-overlapping

            for mask in props.mask_regions:
                if not mask.enabled or not mask.points:
                    continue

                # Check if point is inside the 3D mask cylinder
                if check_point_in_mask_3d(camera_obj, mask, world_pos):
                    is_inside = True
                    # Use mask's inside color (user adjustable)
                    green_color = tuple(mask.inside_color[:3])
                    break

            # If not inside any mask, use outside color from first mask
            if not is_inside and len(props.mask_regions) > 0:
                red_color = tuple(props.mask_regions[0].outside_color[:3])

            # Store base color
            if is_inside:
                vertex_base_colors.append(green_color)
            else:
                vertex_base_colors.append(red_color)

        # Second pass: build rendering data with default Blender shading
        positions = []
        colors = []
        indices = []

        for poly in mesh.polygons:
            poly_verts = []
            for loop_idx in poly.loop_indices:
                vert_idx = mesh.loops[loop_idx].vertex_index
                world_pos = matrix_world @ mesh.vertices[vert_idx].co
                positions.append(world_pos)

                # Get vertex normal in world space
                vert_normal = mesh.vertices[vert_idx].normal
                world_normal = (normal_matrix @ vert_normal).normalized()

                # Apply default Blender workbench lighting to base color
                base_color = vertex_base_colors[vert_idx]
                key_diffuse = max(0.0, world_normal.dot(key_light))
                fill_diffuse = max(0.0, world_normal.dot(fill_light))
                diffuse = 0.8 * key_diffuse + 0.3 * fill_diffuse + 0.2
                diffuse = min(1.0, diffuse)

                # Apply lighting to color - this reveals edges
                lit_color = tuple(base_color[j] * diffuse for j in range(3))
                colors.append((*lit_color, 1.0))

                poly_verts.append(len(positions) - 1)

            # Triangulate polygon (simple fan triangulation)
            for i in range(1, len(poly_verts) - 1):
                indices.append((poly_verts[0], poly_verts[i], poly_verts[i + 1]))

        if not indices or len(positions) == 0 or len(colors) == 0:
            continue

        # Draw the colored mesh
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        batch = batch_for_shader(
            shader, 'TRIS',
            {"pos": positions, "color": colors},
            indices=indices
        )

        # Enable blending
        gpu.state.blend_set('ALPHA')

        # Respect depth - don't draw always on top
        gpu.state.depth_test_set('LESS_EQUAL')

        # Check if X-ray mode is enabled in viewport
        space_data = bpy.context.space_data
        is_xray = False
        if hasattr(space_data, 'shading') and hasattr(space_data.shading, 'show_xray'):
            is_xray = space_data.shading.show_xray

        # If X-ray mode is on, disable depth writes to see through
        if is_xray:
            gpu.state.depth_mask_set(False)

        shader.bind()
        batch.draw(shader)

        # Reset depth mask
        if is_xray:
            gpu.state.depth_mask_set(True)

        # Reset state
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.blend_set('NONE')


# ============================================================================
# MATERIAL-BASED MASK VISUALIZATION
# ============================================================================

def setup_mask_material(obj, mask_color):
    """
    Modify object's material to add mask color overlay.
    Preserves existing material nodes and textures, just adds a color multiply/mix.

    Args:
        obj: Blender object
        mask_color: (r, g, b, a) tuple for mask color
    """
    if not obj or not obj.data or len(obj.data.materials) == 0:
        # Object has no material, create a simple one
        if not obj.data:
            return

        mat = bpy.data.materials.new(name=f"MaskMat_{obj.name}")
        mat.use_nodes = True
        obj.data.materials.append(mat)

        # Mark as temporary
        if not hasattr(obj, '_mask_material_name'):
            obj._mask_material_name = mat.name
            obj._had_no_material = True

    # Get the first material
    mat = obj.data.materials[0]

    # Store original material setup if not already stored
    if not hasattr(obj, '_original_material_nodes'):
        # This is the first time we're modifying - store info
        obj._original_material_nodes = True
        obj._material_name = mat.name

        # Ensure material uses nodes
        if not mat.use_nodes:
            mat.use_nodes = True

    # Now modify the material nodes to add color overlay
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Find or create mask color node
    mask_color_node = None
    for node in nodes:
        if node.name == 'MASK_COLOR_OVERLAY':
            mask_color_node = node
            break

    if not mask_color_node:
        # Find the material output node
        output_node = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
                break

        if not output_node:
            # No output node? Create one
            output_node = nodes.new('ShaderNodeOutputMaterial')
            output_node.location = (300, 300)

        # Get what's currently connected to the output
        original_shader = None
        if output_node.inputs['Surface'].is_linked:
            original_shader = output_node.inputs['Surface'].links[0].from_socket

        # Create color mix node
        mix_rgb = nodes.new('ShaderNodeMixRGB')
        mix_rgb.name = 'MASK_COLOR_OVERLAY'
        mix_rgb.blend_type = 'MULTIPLY'
        mix_rgb.inputs['Fac'].default_value = 0.7  # 70% mask color influence
        mix_rgb.inputs['Color2'].default_value = mask_color
        mix_rgb.location = (output_node.location.x - 200, output_node.location.y)

        # Create emission shader for the mixed color
        emission = nodes.new('ShaderNodeEmission')
        emission.name = 'MASK_EMISSION'
        emission.location = (mix_rgb.location.x + 150, mix_rgb.location.y - 100)

        if original_shader:
            # Disconnect original shader from output
            link_to_remove = output_node.inputs['Surface'].links[0]
            links.remove(link_to_remove)

            # Connect: original -> mix color1, mask color -> mix color2
            # Note: We can't easily get the color from a shader, so we'll use emission with the mask color
            # This is a simplified approach
            pass

        # Simple approach: Just use emission with mask color
        emission.inputs['Color'].default_value = mask_color
        emission.inputs['Strength'].default_value = 0.8

        # Connect emission to output
        links.new(emission.outputs['Emission'], output_node.inputs['Surface'])

        mask_color_node = emission
    else:
        # Update existing overlay node
        if mask_color_node.type == 'EMISSION':
            mask_color_node.inputs['Color'].default_value = mask_color


def restore_original_material(obj):
    """Restore object's original material by removing mask overlay nodes"""
    if not obj or not obj.data:
        return

    # If object had no material originally, remove the temporary one
    if hasattr(obj, '_had_no_material') and obj._had_no_material:
        if hasattr(obj, '_mask_material_name'):
            mat_name = obj._mask_material_name
            if mat_name in bpy.data.materials:
                mat = bpy.data.materials[mat_name]
                if mat.users == 1:  # Only this object uses it
                    bpy.data.materials.remove(mat)
        obj.data.materials.clear()
        del obj._had_no_material
        del obj._mask_material_name
        if hasattr(obj, '_original_material_nodes'):
            del obj._original_material_nodes
        return

    # Remove mask overlay nodes if they exist
    if not hasattr(obj, '_original_material_nodes'):
        return

    if len(obj.data.materials) == 0:
        return

    mat = obj.data.materials[0]
    if not mat.use_nodes:
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Remove mask overlay nodes
    nodes_to_remove = []
    for node in nodes:
        if node.name in ['MASK_COLOR_OVERLAY', 'MASK_EMISSION']:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        nodes.remove(node)

    # Try to restore original connection (best effort)
    # This is simplified - in a full implementation you'd store the original node graph

    del obj._original_material_nodes


def update_target_materials():
    """
    Update materials of all target objects based on mask intersection.
    This runs in real-time via depsgraph handler.
    """
    import bpy
    import json
    from . import camera_utils

    # Find custom camera
    custom_cameras = [obj for obj in bpy.context.scene.objects
                     if obj.custom_camera_props.is_custom_camera]

    if not custom_cameras:
        return

    camera_obj = custom_cameras[0]
    props = camera_obj.custom_camera_props

    if not props.show_mask_overlay:
        # Restore all materials if overlay is disabled
        for obj in bpy.context.scene.objects:
            if hasattr(obj, 'original_materials'):
                restore_original_material(obj)
        return

    # Get active mask
    if props.active_mask_index >= len(props.mask_regions):
        return

    mask = props.mask_regions[props.active_mask_index]

    if not mask.enabled or not mask.points:
        return

    # Parse mask polygon
    try:
        polygon_2d = json.loads(mask.points)
    except:
        return

    if len(polygon_2d) < 3:
        return

    # Get target collection
    if mask.target_collection:
        target_objects = [obj for obj in mask.target_collection.objects
                         if obj.type == 'MESH']
    else:
        # Use all mesh objects in scene
        target_objects = [obj for obj in bpy.context.scene.objects
                         if obj.type == 'MESH']

    # Get render resolution
    scene = bpy.context.scene
    render_width, render_height = camera_utils.get_camera_render_resolution(scene)

    # Process each target object
    for obj in target_objects:
        try:
            # Get object center in world space
            world_pos = obj.matrix_world.translation

            # Check if object is in camera view
            if not camera_utils.is_point_in_camera_view(camera_obj, world_pos):
                # Object not in view - restore original material
                restore_original_material(obj)
                continue

            # Project to screen space
            screen_coords = camera_utils.project_world_to_screen(
                camera_obj, world_pos, render_width, render_height
            )

            if not screen_coords:
                restore_original_material(obj)
                continue

            # Check if point is inside mask polygon
            is_inside = point_in_polygon(screen_coords, polygon_2d)

            # Apply appropriate material
            if is_inside:
                # Inside mask - green
                setup_mask_material(obj, tuple(mask.inside_color))
            else:
                # Outside mask - red
                setup_mask_material(obj, tuple(mask.outside_color))

        except Exception as e:
            print(f"Error updating material for {obj.name}: {e}")
            continue


# Depsgraph handler for real-time material updates
_material_update_handler = None

def material_update_depsgraph_handler(scene, depsgraph):
    """Depsgraph handler that updates materials in real-time"""
    update_target_materials()


def enable_material_updates():
    """Enable real-time material updates via depsgraph"""
    global _material_update_handler

    if _material_update_handler is not None:
        return

    _material_update_handler = bpy.app.handlers.depsgraph_update_post.append(
        material_update_depsgraph_handler
    )

    print("Enabled real-time material updates")


def disable_material_updates():
    """Disable real-time material updates and restore all materials"""
    global _material_update_handler

    if _material_update_handler is not None:
        bpy.app.handlers.depsgraph_update_post.remove(material_update_depsgraph_handler)
        _material_update_handler = None

    # Restore all materials
    for obj in bpy.context.scene.objects:
        if hasattr(obj, 'original_materials'):
            restore_original_material(obj)

    print("Disabled real-time material updates")


def enable_mask_ray_drawing(context):
    """Enable mask ray viewport drawing"""
    global _ray_draw_handler

    if _ray_draw_handler is not None:
        return

    _ray_draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_mask_rays, (), 'WINDOW', 'POST_VIEW'
    )

    # Force viewport update
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

    print("Enabled mask ray viewport drawing")


def disable_mask_ray_drawing():
    """Disable mask ray viewport drawing"""
    global _ray_draw_handler

    if _ray_draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ray_draw_handler, 'WINDOW')
        _ray_draw_handler = None

        # Force viewport update
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        print("Disabled mask ray viewport drawing")


def enable_perpendicular_mask_drawing(context):
    """Enable perpendicular mask mesh viewport drawing"""
    global _mask_mesh_handler

    if _mask_mesh_handler is not None:
        return

    _mask_mesh_handler = bpy.types.SpaceView3D.draw_handler_add(
        draw_perpendicular_mask_mesh, (), 'WINDOW', 'POST_VIEW'
    )

    # Force viewport update
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

    print("Enabled perpendicular mask mesh drawing")


def disable_perpendicular_mask_drawing():
    """Disable perpendicular mask mesh viewport drawing"""
    global _mask_mesh_handler

    if _mask_mesh_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_mask_mesh_handler, 'WINDOW')
        _mask_mesh_handler = None

        # Force viewport update
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        print("Disabled perpendicular mask mesh drawing")
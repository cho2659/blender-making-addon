import bpy
import gpu
import json
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Matrix
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
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y
    
    screen_pos = camera_utils.project_world_to_screen(
        camera_obj, world_pos, resolution_x, resolution_y
    )
    
    if screen_pos is None:
        return False
    
    # Check if projected point is inside polygon
    return point_in_polygon(screen_pos, points_2d)


def _mask_targets_object(mask, target_obj):
    """Return True if mask should consider the given object"""
    if not target_obj:
        return False

    if hasattr(mask, "target_collection") and mask.target_collection:
        try:
            return target_obj in mask.target_collection.objects
        except Exception:
            return False

    # No explicit collection target - apply to everything
    return True


def _get_applicable_masks(props, target_obj):
    """Gather enabled masks that apply to the target object"""
    applicable = []

    for mask in getattr(props, "mask_regions", []):
        if not mask.enabled or not mask.points:
            continue
        if _mask_targets_object(mask, target_obj):
            applicable.append(mask)

    if applicable:
        return applicable

    # Fallback: allow any enabled mask if no specific targets were matched
    for mask in getattr(props, "mask_regions", []):
        if mask.enabled and mask.points:
            applicable.append(mask)

    return applicable


def _average_mask_color(samples):
    """Average a list of RGB tuples"""
    if not samples:
        return (0.0, 0.0, 0.0)

    total = [0.0, 0.0, 0.0]
    for color in samples:
        total[0] += color[0]
        total[1] += color[1]
        total[2] += color[2]

    count = float(len(samples))
    return (total[0] / count, total[1] / count, total[2] / count)


def _extract_rgb(color):
    """Ensure color is returned as RGB tuple"""
    if not color:
        return (1.0, 0.0, 0.0)

    if len(color) >= 3:
        return tuple(color[:3])

    # Fallback if Blender returns scalar
    return (color, color, color)


def build_mask_overlay_geometry(camera_obj, props, target_obj):
    """
    Build face-based overlay geometry for the given target object.
    Returns (positions, colors, indices) suitable for GPU drawing.
    """
    if not camera_obj or not target_obj or target_obj.type != 'MESH':
        return None

    mesh = target_obj.data
    if not mesh or len(mesh.polygons) == 0:
        return None

    applicable_masks = _get_applicable_masks(props, target_obj)
    if not applicable_masks:
        return None

    matrix_world = target_obj.matrix_world

    try:
        normal_matrix = matrix_world.to_3x3().inverted().transposed()
    except Exception:
        normal_matrix = Matrix.Identity(3)

    # Determine colors
    fallback_color = _extract_rgb(applicable_masks[0].outside_color
                                  if applicable_masks else (1.0, 0.0, 0.0))

    # Lighting vectors roughly matching Blender's workbench light
    key_light = Vector((0.6, 0.4, -0.7)).normalized()
    fill_light = Vector((-0.3, 0.2, -0.6)).normalized()

    dims = target_obj.dimensions
    max_dim = max(dims) if dims else 1.0
    surface_offset = max(0.0005, max_dim * 0.001)

    positions = []
    colors = []
    indices = []

    for poly in mesh.polygons:
        if len(poly.loop_indices) < 3:
            continue

        world_positions = []
        world_normals = []
        loop_indices = []

        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            vert = mesh.vertices[vert_idx]
            world_pos = matrix_world @ vert.co
            world_positions.append(world_pos)

            vert_normal = vert.normal
            world_normal = (normal_matrix @ vert_normal).normalized()
            world_normals.append(world_normal)
            loop_indices.append(loop_idx)

        sample_points = list(world_positions)
        if len(world_positions) >= 3:
            centroid = Vector((0.0, 0.0, 0.0))
            for pos in world_positions:
                centroid += pos
            centroid /= len(world_positions)
            sample_points.append(centroid)

        for i in range(len(world_positions)):
            next_idx = (i + 1) % len(world_positions)
            edge_mid = (world_positions[i] + world_positions[next_idx]) * 0.5
            sample_points.append(edge_mid)

        inside_samples = []
        for sample in sample_points:
            for mask in applicable_masks:
                if check_point_in_mask_3d(camera_obj, mask, sample):
                    inside_samples.append(_extract_rgb(mask.inside_color))
                    break

        coverage = (len(inside_samples) / len(sample_points)) if sample_points else 0.0
        coverage = max(0.0, min(1.0, coverage))

        if inside_samples:
            inside_color = _average_mask_color(inside_samples)
            # Blend to keep some hint of outside color for readability
            face_color = tuple(
                fallback_color[i] * (1.0 - coverage) + inside_color[i] * coverage
                for i in range(3)
            )
            alpha = 0.3 + 0.6 * coverage
        else:
            face_color = fallback_color
            alpha = 0.12

        alpha = max(0.05, min(0.65, alpha))

        poly_indices = []
        for idx, world_pos in enumerate(world_positions):
            offset_pos = world_pos + world_normals[idx] * surface_offset
            positions.append(offset_pos)

            world_normal = world_normals[idx]
            key_diffuse = max(0.0, world_normal.dot(key_light))
            fill_diffuse = max(0.0, world_normal.dot(fill_light))
            diffuse = min(1.0, 0.8 * key_diffuse + 0.3 * fill_diffuse + 0.2)

            lit_color = tuple(face_color[j] * diffuse for j in range(3))
            colors.append((*lit_color, alpha))
            poly_indices.append(len(positions) - 1)

        for i in range(1, len(poly_indices) - 1):
            indices.append((poly_indices[0], poly_indices[i], poly_indices[i + 1]))

    if not positions or not indices:
        return None

    return positions, colors, indices

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
    
    # Draw overlay using the general utility so rendering rules stay unified
    draw_target_object_coloring(camera_obj, props, override_objects=target_objects)


def draw_object_overlay(camera_obj, target_obj, props):
    """Draw overlay for a single target object with depth-aware blending"""
    geometry = build_mask_overlay_geometry(camera_obj, props, target_obj)
    if not geometry:
        return

    positions, colors, indices = geometry
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": positions, "color": colors},
        indices=indices
    )

    space_data = getattr(bpy.context, "space_data", None)
    is_xray = False
    if space_data and hasattr(space_data, 'shading') and hasattr(space_data.shading, 'show_xray'):
        is_xray = space_data.shading.show_xray

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('ALWAYS' if is_xray else 'LESS_EQUAL')
    gpu.state.depth_mask_set(False)

    shader.bind()
    batch.draw(shader)

    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set('LESS_EQUAL')
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


def draw_target_object_coloring(camera_obj, props, override_objects=None):
    """
    Color target objects based on mask regions using face-based sampling so entire faces highlight.
    The overlay is rendered via GPU blending so it sits in a separate compositing pass.
    """
    # Determine which objects to draw
    if override_objects is not None:
        target_objects = [obj for obj in override_objects if obj and obj.type == 'MESH']
    else:
        target_objects = []

        for mask in props.mask_regions:
            if not mask.enabled or not mask.target_collection:
                continue

            for obj in mask.target_collection.objects:
                if obj.type == 'MESH' and obj not in target_objects:
                    target_objects.append(obj)

        if not target_objects and hasattr(props, 'target_objects'):
            for target_item in props.target_objects:
                if target_item.obj and target_item.obj.type == 'MESH':
                    target_objects.append(target_item.obj)

    if not target_objects:
        return

    for target_obj in target_objects:
        draw_object_overlay(camera_obj, target_obj, props)


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

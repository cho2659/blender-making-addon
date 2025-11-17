import bpy
from bpy.types import Panel, Operator

# ============================================================================
# PANEL CLASS
# ============================================================================

class CAMERA_PT_custom_camera_panel(Panel):
    """Main panel for Custom Camera controls"""
    bl_label = "Custom Camera Masking"
    bl_idname = "CAMERA_PT_custom_camera_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Custom Camera'
    
    def draw(self, context):
        layout = self.layout
        
        # Find custom camera in scene
        custom_cameras = [obj for obj in context.scene.objects 
                         if obj.custom_camera_props.is_custom_camera]
        
        if not custom_cameras:
            # No custom camera exists
            layout.label(text="No Custom Camera", icon='INFO')
            layout.operator("camera.add_custom_camera", icon='ADD')
            return
        
        camera_obj = custom_cameras[0]
        props = camera_obj.custom_camera_props
        
        # Camera info (removed active camera UI - use scene collection widget instead)
        box = layout.box()
        box.label(text=f"Camera: {camera_obj.name}", icon='CAMERA_DATA')

        layout.separator()
        
        # ====================================================================
        # REFERENCE IMAGE SECTION
        # ====================================================================
        box = layout.box()
        box.label(text="Reference Image", icon='IMAGE_DATA')
        
        # Load image button
        box.operator("camera.load_reference_image", 
                    text="Load Image", icon='FILEBROWSER')
        
        # Show image path if loaded
        if props.reference_image_path:
            col = box.column(align=True)
            col.scale_y = 0.8
            
            # Split path for display
            import os
            filename = os.path.basename(props.reference_image_path)
            col.label(text=f"File: {filename}", icon='FILE_IMAGE')
        
        # Opacity slider (only if image loaded)
        if props.reference_image_path:
            box.prop(props, "reference_image_opacity", text="Opacity", slider=True)
        
        layout.separator()
        
        # ====================================================================
        # MASK REGIONS SECTION
        # ====================================================================
        box = layout.box()
        box.label(text="Mask Regions", icon='MOD_MASK')
        
        # Add/Remove buttons
        row = box.row(align=True)
        row.operator("camera.add_mask_region", text="Add", icon='ADD')
        if len(props.mask_regions) > 0:
            row.operator("camera.remove_mask_region", text="Remove", icon='REMOVE')
        
        # List existing masks
        if len(props.mask_regions) > 0:
            box.separator()
            
            for i, mask in enumerate(props.mask_regions):
                mask_box = box.box()
                
                # Mask header with enable toggle
                row = mask_box.row()
                row.prop(mask, "enabled", text="")
                
                # Make mask active when clicked
                op = row.operator("camera.set_active_mask", text=mask.name, 
                                 emboss=False)
                op.index = i
                
                # Show active indicator
                if i == props.active_mask_index:
                    row.label(text="", icon='RADIOBUT_ON')
                else:
                    row.label(text="", icon='RADIOBUT_OFF')
                
                # Only show details for active mask
                if i == props.active_mask_index:
                    col = mask_box.column(align=True)

                    # Draw mask button
                    if mask.points:
                        col.operator("camera.draw_mask",
                                   text="Redraw Mask", icon='GREASEPENCIL')

                        # Show point count
                        import json
                        try:
                            points = json.loads(mask.points)
                            col.label(text=f"Points: {len(points)}", icon='PIVOT_INDIVIDUAL')
                        except:
                            pass
                    else:
                        col.operator("camera.draw_mask",
                                   text="Draw Mask", icon='GREASEPENCIL')

                    col.separator()

                    # Mask mode and collection
                    col.label(text="Mask Settings:", icon='SETTINGS')
                    col.prop(mask, "mask_mode", text="Mode")
                    col.prop(mask, "target_collection", text="Collection")

                    col.separator()

                    # Colors
                    col.prop(mask, "inside_color", text="Inside")
                    col.prop(mask, "outside_color", text="Outside")

                    col.separator()

                    # 3D Cylinder Mesh Settings
                    col.label(text="3D Cylinder Mesh:", icon='MESH_CYLINDER')
                    col.prop(mask, "fade_start", text="Start Distance")
                    col.prop(mask, "fade_end", text="End Distance")
                    col.prop(mask, "ray_opacity", text="Opacity", slider=True)

                    col.separator()

                    # Edge detection settings
                    col.label(text="Edge Detection:", icon='HAND')
                    col.prop(mask, "use_auto_refine", text="Auto-Refine")
                    if mask.use_auto_refine:
                        col.prop(mask, "edge_sensitivity", text="Sensitivity", slider=True)
                        col.prop(mask, "detail_level", text="Detail", slider=True)
                        col.prop(mask, "contour_constraint", text="Follow Drawing", slider=True)
        else:
            box.label(text="No masks", icon='INFO')

        layout.separator()
        
        # ====================================================================
        # VISUALIZATION SECTION
        # ====================================================================
        box = layout.box()
        box.label(text="Visualization", icon='SHADING_RENDERED')
        
        # Show overlay toggle
        box.prop(props, "show_mask_overlay", text="Show Mask Overlay", toggle=True)
        
        # Intersection color
        if props.blend_mode == 'INTERSECTION':
            box.prop(props, "intersection_color", text="Intersection")


# ============================================================================
# OPERATOR CLASSES
# ============================================================================

class CAMERA_OT_activate_custom_camera(Operator):
    """Set custom camera as active scene camera"""
    bl_idname = "camera.activate_custom_camera"
    bl_label = "Activate Custom Camera"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects 
                         if obj.custom_camera_props.is_custom_camera]
        
        if custom_cameras:
            context.scene.camera = custom_cameras[0]
            
            # Switch to camera view
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.region_3d.view_perspective = 'CAMERA'
                            break
            
            self.report({'INFO'}, "Activated custom camera")
        
        return {'FINISHED'}


class CAMERA_OT_toggle_camera_visibility(Operator):
    """Toggle camera visibility in viewport"""
    bl_idname = "camera.toggle_camera_visibility"
    bl_label = "Toggle Camera Visibility"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects
                         if obj.custom_camera_props.is_custom_camera]

        if custom_cameras:
            camera_obj = custom_cameras[0]
            camera_obj.hide_viewport = not camera_obj.hide_viewport

            if camera_obj.hide_viewport:
                self.report({'INFO'}, "Camera hidden in viewport")
            else:
                self.report({'INFO'}, "Camera visible in viewport")

        return {'FINISHED'}


class CAMERA_OT_set_active_mask(Operator):
    """Set active mask region"""
    bl_idname = "camera.set_active_mask"
    bl_label = "Set Active Mask"
    bl_options = {'REGISTER', 'UNDO'}

    index: bpy.props.IntProperty()

    def execute(self, context):
        custom_cameras = [obj for obj in context.scene.objects
                         if obj.custom_camera_props.is_custom_camera]

        if custom_cameras:
            camera_obj = custom_cameras[0]
            props = camera_obj.custom_camera_props
            if 0 <= self.index < len(props.mask_regions):
                props.active_mask_index = self.index

                # Switch camera background to the selected mask's overlay
                selected_mask = props.mask_regions[self.index]
                if selected_mask.mask_overlay_path:
                    # Load the mask overlay image
                    try:
                        if selected_mask.mask_overlay_path in bpy.data.images:
                            mask_img = bpy.data.images[selected_mask.mask_overlay_path]
                            mask_img.reload()
                        else:
                            mask_img = bpy.data.images.load(selected_mask.mask_overlay_path)

                        # Update camera background
                        from . import camera_utils
                        camera_utils.setup_camera_background(
                            camera_obj, mask_img, props.reference_image_opacity
                        )
                        print(f"Switched to mask overlay: {selected_mask.mask_overlay_path}")
                    except Exception as e:
                        print(f"Could not load mask overlay: {e}")
                        # Fall back to original image if mask overlay fails
                        if props.reference_image_path:
                            try:
                                orig_img = bpy.data.images.load(props.reference_image_path)
                                from . import camera_utils
                                camera_utils.setup_camera_background(
                                    camera_obj, orig_img, props.reference_image_opacity
                                )
                            except:
                                pass
                else:
                    # No overlay for this mask, show original image
                    if props.reference_image_path:
                        try:
                            if props.reference_image_path in bpy.data.images:
                                orig_img = bpy.data.images[props.reference_image_path]
                                orig_img.reload()
                            else:
                                orig_img = bpy.data.images.load(props.reference_image_path)

                            from . import camera_utils
                            camera_utils.setup_camera_background(
                                camera_obj, orig_img, props.reference_image_opacity
                            )
                            print(f"Switched to original image (no mask overlay)")
                        except Exception as e:
                            print(f"Could not load original image: {e}")

        return {'FINISHED'}


# ============================================================================
# NO REGISTRATION HERE!
# Registration is handled in __init__.py
# ============================================================================
# 
# This file ONLY contains class definitions.
# The classes are imported and registered in __init__.py
#
import bpy
from bpy.props import (
    StringProperty, 
    BoolProperty, 
    FloatProperty, 
    CollectionProperty,
    PointerProperty,
    FloatVectorProperty,
    IntProperty
)

def update_opacity_callback(self, context):
    """Update callback when opacity slider changes"""
    obj = context.object
    if obj and obj.type == 'CAMERA' and obj.data.background_images:
        for bg in obj.data.background_images:
            bg.alpha = self.reference_image_opacity

def update_show_mask_overlay_callback(self, context):
    """Update callback when show_mask_overlay toggle changes"""
    from . import camera_utils

    # Find custom camera
    custom_cameras = [obj for obj in context.scene.objects
                     if hasattr(obj, 'custom_camera_props') and obj.custom_camera_props.is_custom_camera]

    if not custom_cameras:
        return

    camera_obj = custom_cameras[0]
    props = camera_obj.custom_camera_props

    if self.show_mask_overlay:
        # Show mask overlay if available
        if props.active_mask_index < len(props.mask_regions):
            active_mask = props.mask_regions[props.active_mask_index]
            if active_mask.mask_overlay_path:
                try:
                    if active_mask.mask_overlay_path in bpy.data.images:
                        mask_img = bpy.data.images[active_mask.mask_overlay_path]
                        mask_img.reload()
                    else:
                        mask_img = bpy.data.images.load(active_mask.mask_overlay_path)

                    camera_utils.setup_camera_background(
                        camera_obj, mask_img, props.reference_image_opacity
                    )
                    print("Showing mask overlay")
                except Exception as e:
                    print(f"Could not load mask overlay: {e}")
    else:
        # Show original image
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
                print("Showing original image")
            except Exception as e:
                print(f"Could not load original image: {e}")

class TargetObject(bpy.types.PropertyGroup):
    """Individual target object for masking"""
    obj: PointerProperty(
        name="Object",
        type=bpy.types.Object,
        description="Target object to check against masks"
    )

class MaskRegion(bpy.types.PropertyGroup):
    """Individual mask region definition"""
    name: StringProperty(
        name="Mask Name",
        default="Mask1"
    )
    
    # Store polygon points for the mask outline
    points: StringProperty(
        name="Points Data",
        description="Serialized polygon points",
        default=""
    )

    # Store path to mask overlay image (so we can switch between masks)
    mask_overlay_path: StringProperty(
        name="Mask Overlay Path",
        description="Path to temporary mask overlay image",
        default=""
    )

    # Auto-detection settings
    detection_mode: bpy.props.EnumProperty(
        name="Detection Mode",
        items=[
            ('MANUAL', "Manual", "Draw mask manually"),
            ('EDGE', "Edge Detection", "Detect edges automatically"),
            ('COLOR', "Color Pick", "Use color similarity"),
            ('ALPHA', "Alpha Channel", "Use image transparency"),
        ],
        default='ALPHA',
        description="How to detect the mask region"
    )
    
    pick_color: FloatVectorProperty(
        name="Pick Color",
        subtype='COLOR',
        default=(1.0, 1.0, 1.0),
        size=3,
        min=0.0,
        max=1.0,
        description="Color to detect (for COLOR mode)"
    )
    
    color_tolerance: FloatProperty(
        name="Color Tolerance",
        default=0.1,
        min=0.0,
        max=1.0,
        description="How similar colors must be"
    )
    
    alpha_threshold: FloatProperty(
        name="Alpha Threshold",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Alpha value threshold for detection"
    )
    
    edge_threshold: FloatProperty(
        name="Edge Threshold",
        default=0.3,
        min=0.0,
        max=1.0,
        description="Edge detection sensitivity"
    )

    # GrabCut constraint - controls mask refinement tightness
    contour_constraint: FloatProperty(
        name="Follow User Drawing",
        default=0.015,
        min=0.001,
        max=0.5,
        description="Controls mask refinement: lower = follows exact object edges, higher = stays closer to your drawing"
    )

    # Visual properties
    inside_color: FloatVectorProperty(
        name="Inside Color",
        subtype='COLOR',
        default=(0.0, 1.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        description="Color when object is inside mask"
    )
    
    outside_color: FloatVectorProperty(
        name="Outside Color",
        subtype='COLOR',
        default=(1.0, 0.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        description="Color when object is outside mask"
    )
    
    # Depth fade parameters
    fade_start: FloatProperty(
        name="Fade Start",
        default=1.0,
        min=0.0,
        description="Distance from camera where mask cylinder starts (sensor plane distance)"
    )
    
    fade_end: FloatProperty(
        name="Fade End",
        default=10.0,
        min=0.0,
        description="Distance from camera where mask is fully transparent"
    )
    
    transparency_at_camera: FloatProperty(
        name="Transparency at Camera",
        default=0.0,
        min=0.0,
        max=1.0,
        description="Transparency level at camera position"
    )
    
    enabled: BoolProperty(
        name="Enabled",
        default=True,
        description="Enable/disable this mask region"
    )

    # Mask mode (only target mode supported)
    mask_mode: bpy.props.EnumProperty(
        name="Mask Mode",
        items=[
            ('TARGET', "Target Mask", "Objects in this mask are targets"),
        ],
        default='TARGET',
        description="How this mask affects objects"
    )

    # Collection for this mask
    target_collection: PointerProperty(
        name="Target Collection",
        type=bpy.types.Collection,
        description="Collection to apply this mask to"
    )

    # Ray visualization settings
    show_rays: BoolProperty(
        name="Show Rays",
        default=True,
        description="Show 3D rays in viewport"
    )

    ray_opacity: FloatProperty(
        name="Ray Opacity",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Opacity of the mask rays"
    )

    ray_length: FloatProperty(
        name="Ray Length",
        default=10.0,
        min=0.1,
        max=100.0,
        description="Length of the mask rays in Blender units"
    )

    # Edge detection refinement settings
    use_auto_refine: BoolProperty(
        name="Auto-Refine Mask",
        default=True,
        description="Automatically refine mask using edge detection"
    )

    edge_sensitivity: FloatProperty(
        name="Edge Sensitivity",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Sensitivity of edge detection (higher = more sensitive)"
    )

    detail_level: FloatProperty(
        name="Detail Level",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Level of detail in mask outline (higher = more points)"
    )

class CustomCameraProperties(bpy.types.PropertyGroup):
    """Properties for Custom Camera object"""
    
    is_custom_camera: BoolProperty(
        name="Is Custom Camera",
        default=False,
        description="Mark this object as a custom camera"
    )
    
    reference_image_path: StringProperty(
        name="Reference Image",
        #subtype='FILE_PATH',
        description="Path to reference image for masking"
    )

    fitted_image_path: StringProperty(
        name="Fitted Image Path",
        description="Path to temporary fitted image"
    )

    # Store fit_info as individual properties for persistence
    fit_original_width: IntProperty(default=0)
    fit_original_height: IntProperty(default=0)
    fit_render_width: IntProperty(default=0)
    fit_render_height: IntProperty(default=0)
    fit_fitted_width: IntProperty(default=0)
    fit_fitted_height: IntProperty(default=0)
    fit_offset_x: IntProperty(default=0)
    fit_offset_y: IntProperty(default=0)
    fit_scale_factor: FloatProperty(default=1.0)

    reference_image_opacity: FloatProperty(
        name="Image Opacity",
        default=0.5,
        min=0.0,
        max=1.0,
        description="Opacity of reference image in camera view",
        update=lambda self, context: update_opacity_callback(self, context)
    )
    
    mask_regions: CollectionProperty(
        type=MaskRegion,
        name="Mask Regions"
    )
    
    active_mask_index: IntProperty(
        name="Active Mask",
        default=0
    )
    
    target_object: PointerProperty(
        name="Target Object",
        type=bpy.types.Object,
        description="Object to check against masks (legacy - use target_objects list)"
    )
    
    target_objects: CollectionProperty(
        type=TargetObject,
        name="Target Objects",
        description="List of target objects to check against masks"
    )
    
    active_target_index: IntProperty(
        name="Active Target",
        default=0,
        description="Index of active target in the list"
    )
    
    show_mask_overlay: BoolProperty(
        name="Show Mask Overlay",
        default=True,
        description="Display mask visualization in viewport",
        update=lambda self, context: update_show_mask_overlay_callback(self, context)
    )
    
    mask_resolution: IntProperty(
        name="Mask Resolution",
        default=512,
        min=64,
        max=2048,
        description="Resolution for mask calculation"
    )
    
    # Multi-camera blending
    blend_mode: bpy.props.EnumProperty(
        name="Blend Mode",
        items=[
            ('SINGLE', "Single", "Use only this camera's mask"),
            ('INTERSECTION', "Intersection", "Show special color when object is in multiple masks"),
            ('UNION', "Union", "Combine multiple camera masks"),
        ],
        default='SINGLE'
    )
    
    intersection_color: FloatVectorProperty(
        name="Intersection Color",
        subtype='COLOR',
        default=(1.0, 1.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        description="Color when object is inside multiple camera masks"
    )
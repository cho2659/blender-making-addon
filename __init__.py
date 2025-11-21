bl_info = {
    "name": "Custom Camera Masking System",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Add > Camera > Custom Camera",
    "description": "Advanced camera system with image-based 3D masking",
    "category": "Camera",
}

import bpy
import sys

# ============================================================================
# CRITICAL: OpenCV Import with Error Handling
# ============================================================================

opencv_available = False
opencv_error = None

try:
    import cv2
    import numpy as np
    # Removed: active_contour and scipy - replaced with faster GrabCut algorithm
    opencv_available = True
    print("✓ OpenCV and NumPy loaded successfully")
except ImportError as e:
    opencv_error = f"Required libraries not installed: {e}\nPlease install: pip install opencv-python numpy"
    print(f"✗ {opencv_error}")
except Exception as e:
    opencv_error = f"Library loading error: {e}"
    print(f"✗ {opencv_error}")

# Import addon modules
from . import properties
from . import camera_utils
from . import mask_utils
from . import panels  # Always import panels

# Only import operators if OpenCV is available
if opencv_available:
    from . import operators
else:
    print("⚠ Running in limited mode without OpenCV features")
    operators = None


# ============================================================================
# CLASS REGISTRATION
# ============================================================================

def get_classes():
    """Get classes to register based on OpenCV availability"""
    
    # Always register properties and panels
    base_classes = [
        # Properties (must be registered first!)
        properties.TargetObject,
        properties.MaskRegion,
        properties.CustomCameraProperties,
    ]
    
    # Panel classes (always register)
    panel_classes = [
        panels.CAMERA_PT_custom_camera_panel,
        panels.CAMERA_OT_activate_custom_camera,
        panels.CAMERA_OT_toggle_camera_visibility,
        panels.CAMERA_OT_set_active_mask,
    ]
    
    # Only add operators if OpenCV is available
    if opencv_available and operators:
        operator_classes = [
            operators.CAMERA_OT_add_custom_camera,
            operators.CAMERA_OT_load_reference_image,
            operators.CAMERA_OT_add_mask_region,
            operators.CAMERA_OT_remove_mask_region,
            operators.CAMERA_OT_set_target_object,
            operators.CAMERA_OT_add_target_object,
            operators.CAMERA_OT_remove_target_object,
            operators.CAMERA_OT_draw_mask,
            operators.CAMERA_OT_cleanup_temp_files,
        ]
        
        return base_classes + operator_classes + panel_classes
    else:
        # Limited mode: only properties and panels
        return base_classes + panel_classes


def menu_func(self, context):
    """Add to Camera menu"""
    if opencv_available and operators:
        self.layout.operator(
            operators.CAMERA_OT_add_custom_camera.bl_idname, 
            icon='OUTLINER_OB_CAMERA'
        )
    else:
        # Show disabled menu item with error message
        layout = self.layout
        layout.enabled = False
        layout.label(text="Custom Camera (OpenCV Required)", icon='ERROR')


def register():
    """Register addon classes and properties"""
    
    print("\n" + "="*60)
    print("Registering Custom Camera Masking System")
    print("="*60)
    
    # Check OpenCV status
    if not opencv_available:
        print(f"\n⚠ WARNING: {opencv_error}")
        print("⚠ Addon will run in LIMITED MODE")
        print("⚠ To enable full features:")
        print("   1. Close Blender completely")
        print("   2. Install: pip install opencv-python numpy scikit-image scipy")
        print("   3. Restart Blender\n")
    
    # Get classes to register
    classes = get_classes()
    
    print(f"\nRegistering {len(classes)} classes:")
    print("-" * 60)
    
    # Register each class
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            print(f"  ✓ {cls.__name__}")
        except Exception as e:
            print(f"  ✗ {cls.__name__}: {e}")
            continue
    
    # Add custom property to all objects
    try:
        bpy.types.Object.custom_camera_props = bpy.props.PointerProperty(
            type=properties.CustomCameraProperties
        )
        print("\n  ✓ Added custom_camera_props to Object")
    except Exception as e:
        print(f"\n  ✗ Failed to add property: {e}")
    
    # Add to menu
    try:
        bpy.types.VIEW3D_MT_camera_add.append(menu_func)
        print("  ✓ Added to Camera menu")
    except Exception as e:
        print(f"  ✗ Failed to add menu: {e}")
    
    print("="*60)
    if opencv_available:
        print("✓ Custom Camera Addon registered successfully!")
        print("  All features enabled")
    else:
        print("⚠ Custom Camera Addon registered in LIMITED MODE")
        print("  Panel visible but operators disabled")
    print("="*60 + "\n")


def unregister():
    """Unregister addon classes and properties"""

    print("\n" + "="*60)
    print("Unregistering Custom Camera Masking System")
    print("="*60)

    # Disable viewport drawing and depsgraph handlers
    try:
        mask_utils.disable_viewport_drawing()
        mask_utils.disable_mask_ray_drawing()
        mask_utils.disable_perpendicular_mask_drawing()
        print("  ✓ Disabled all viewport handlers")
    except Exception as e:
        print(f"  ✗ Failed to disable handlers: {e}")

    # Clear shader cache
    try:
        mask_utils._custom_shader_cache = None
        print("  ✓ Cleared shader cache")
    except Exception as e:
        print(f"  ✗ Failed to clear shader cache: {e}")

    # Remove from menu
    try:
        bpy.types.VIEW3D_MT_camera_add.remove(menu_func)
        print("  ✓ Removed from Camera menu")
    except Exception as e:
        print(f"  ✗ Failed to remove menu: {e}")
    
    # Remove custom property
    try:
        if hasattr(bpy.types.Object, 'custom_camera_props'):
            del bpy.types.Object.custom_camera_props
            print("  ✓ Removed custom_camera_props")
    except Exception as e:
        print(f"  ✗ Failed to remove property: {e}")
    
    # Unregister classes (reverse order)
    classes = get_classes()
    
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
            print(f"  ✓ Unregistered: {cls.__name__}")
        except Exception as e:
            print(f"  ✗ Failed to unregister {cls.__name__}: {e}")
            continue
    
    # Clean up OpenCV on Windows
    if opencv_available:
        try:
            import cv2
            # Force cleanup
            if hasattr(cv2, '__loader__'):
                delattr(cv2, '__loader__')
            
            # Remove from sys.modules
            modules_to_remove = [key for key in sys.modules.keys() 
                               if key.startswith('cv2')]
            for mod in modules_to_remove:
                del sys.modules[mod]
            
            print("  ✓ Cleaned up OpenCV")
        except Exception as e:
            print(f"  ⚠ OpenCV cleanup warning: {e}")
    
    print("="*60)
    print("✓ Custom Camera Addon unregistered!")
    print("="*60 + "\n")


if __name__ == "__main__":
    register()


# ============================================================================
# WHAT THIS FILE DOES
# ============================================================================
"""
REGISTRATION FLOW:

1. Try to import OpenCV
   - Success → opencv_available = True
   - Failure → opencv_available = False (limited mode)

2. Import all modules
   - properties (always)
   - camera_utils (always)
   - mask_utils (always)
   - panels (always)
   - operators (only if OpenCV available)

3. Build class list based on availability
   - Always: properties + panels
   - If OpenCV: add operators

4. Register classes in order
   - Properties first (they're referenced by operators)
   - Operators next
   - Panels last

5. Add custom property to Object
   - Allows all objects to have custom_camera_props

6. Add menu item
   - Enabled if OpenCV available
   - Disabled with error message if not

WHAT'S DIFFERENT FROM BEFORE:

✓ panels.py NO LONGER has its own register/unregister
✓ ALL registration happens here in __init__.py
✓ Panel operators (activate, deactivate, set_active_mask) registered here
✓ No duplicate registration conflicts

CLASSES REGISTERED (with OpenCV):
  1. TargetObject (property)
  2. MaskRegion (property)
  3. CustomCameraProperties (property)
  4. CAMERA_OT_add_custom_camera (operator)
  5. CAMERA_OT_load_reference_image (operator)
  6. CAMERA_OT_add_mask_region (operator)
  7. CAMERA_OT_remove_mask_region (operator)
  8. CAMERA_OT_set_target_object (operator)
  9. CAMERA_OT_add_target_object (operator)
  10. CAMERA_OT_remove_target_object (operator)
  11. CAMERA_OT_draw_mask (operator)
  12. CAMERA_PT_custom_camera_panel (panel)
  13. CAMERA_OT_activate_custom_camera (panel operator)
  14. CAMERA_OT_deactivate_custom_camera (panel operator)
  15. CAMERA_OT_set_active_mask (panel operator)

Total: 15 classes

CLASSES REGISTERED (without OpenCV - LIMITED MODE):
  1-3. Properties (same as above)
  12-15. Panels (same as above)

Total: 7 classes (operators 4-11 skipped)
"""
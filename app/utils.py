import cv2
import numpy as np
import os

def save_segmentation_result(original_img, segmented_img, tumor_mask, result_path, overlay_path):
    """
    Save the segmentation result and create an overlay image
    
    Args:
        original_img: Original input image
        segmented_img: Segmented image with colors for each class
        tumor_mask: Binary mask of tumor regions
        result_path: Path to save the segmentation result
        overlay_path: Path to save the overlay image
    """
    # Ensure all images are in the right format for visualization
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)
        
    # Ensure original image is 3-channel for blending
    if len(original_img.shape) == 2:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        original_rgb = original_img
    
    # Save the pure segmentation result
    cv2.imwrite(result_path, segmented_img)
    
    # Create overlay by blending original image with segmentation
    # Apply segmentation only where tumor is present (using tumor_mask)
    overlay = original_rgb.copy()
    alpha = 0.6  # Transparency factor
    
    # Only apply segmentation overlay where tumor exists
    mask = tumor_mask.astype(bool)
    overlay[mask] = cv2.addWeighted(
        original_rgb[mask], 1-alpha, 
        segmented_img[mask], alpha, 
        0
    )
    
    # Add contour around tumor for better visibility
    contours, _ = cv2.findContours(
        tumor_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    
    # Save the overlay image
    cv2.imwrite(overlay_path, overlay)
    
    return True
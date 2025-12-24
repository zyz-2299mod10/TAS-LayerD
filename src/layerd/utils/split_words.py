import cv2
import numpy as np
import os
import glob

def segment_image(img, bboxes=None, kernel_w=50, kernel_h=5):
    """
    Args:
        img: Input image (RGBA usually).
        bboxes: List of bounding boxes from EasyOCR [[p1, p2, p3, p4], ...].
                If provided, adaptive dilation is used.
        kernel_w: Default horizontal kernel width (fallback).
        kernel_h: Default horizontal kernel height (fallback).
    """
    if img is None:
        print("Error: Input image is None")
        return None, None, None

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    h_img, w_img = img.shape[:2]
    
    # Dilation 
    if bboxes is not None and len(bboxes) > 0:
        # Adaptive Dilation
        final_dilated = np.zeros_like(binary)
        
        for bbox in bboxes:
            # EasyOCR bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            pts = np.array(bbox, dtype=np.int32)
            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])
            
            w_box = x_max - x_min
            h_box = y_max - y_min
            
            if h_box > w_box * 1.5: 
                k_w, k_h = kernel_h, kernel_w 
                padding = 10
            else:
                k_w, k_h = kernel_w, kernel_h
                padding = 10
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
            
            # Padding ROI
            roi_x1 = max(0, x_min - padding)
            roi_y1 = max(0, y_min - padding)
            roi_x2 = min(w_img, x_max + padding)
            roi_y2 = min(h_img, y_max + padding)
            
            roi_binary = binary[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi_binary.size == 0:
                continue

            # partial dilation
            roi_dilated = cv2.dilate(roi_binary, kernel, iterations=1)
            
            current_area = final_dilated[roi_y1:roi_y2, roi_x1:roi_x2]
            final_dilated[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.bitwise_or(current_area, roi_dilated)
            
        dilated = final_dilated
        
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        dilated = cv2.dilate(binary, kernel, iterations=1)

    # Find Contours 
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        contour_list.append((c, y, w, h, x))
    
    # Sort by Y position
    contour_list.sort(key=lambda item: item[1]) 

    # Debug Image Visualization
    white_bg = np.ones((h_img, w_img, 3), dtype=np.uint8) * 255
    if img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        alpha_f = a.astype(float) / 255.0
        foreground = cv2.merge((b, g, r)).astype(float)
        blended = (foreground * alpha_f[..., None]) + (white_bg.astype(float) * (1 - alpha_f[..., None]))
        debug_img = blended.astype(np.uint8)
    else:
        debug_img = img.copy()

    layers = []
    for i, (cnt, y, w, h, x) in enumerate(contour_list):
        if w < 5 or h < 5: 
            continue

        single_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        
        layer_canvas = cv2.bitwise_and(img, img, mask=single_mask)
        layers.append(layer_canvas)
        
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return layers, dilated, debug_img
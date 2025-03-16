import cv2
import numpy as np
from core.config import settings

def draw_dotted_rectangle(img, pt1, pt2, color, gap=5):
    """Draw a dotted rectangle on the image."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw dotted lines for each side
    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y1), (x + gap, y1), color, thickness=1)

    for x in range(x1, x2, gap * 2):
        cv2.line(img, (x, y2), (x + gap, y2), color, thickness=1)

    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x1, y), (x1, y + gap), color, thickness=1)

    for y in range(y1, y2, gap * 2):
        cv2.line(img, (x2, y), (x2, y + gap), color, thickness=1)

def process_yolo_result(image_array, result):
    """Process YOLO detection results and visualize them on the image."""
    # Extract bounding boxes and confidence scores
    boxes = result[0].boxes.xyxy.cpu().numpy()
    confidences = result[0].boxes.conf.cpu().numpy()
    masks = result[0].masks
    locule_count = len(boxes)
    
    # Assign colors to detections
    colors = [settings.COLORS[i % len(settings.COLORS)] for i in range(locule_count)]
    
    # Create an overlay for segmentation masks
    overlay = np.zeros_like(image_array, dtype=np.uint8)
    
    # Process masks if they exist
    if masks is not None:
        mask_tensor = masks.data.cpu()
        num_masks = mask_tensor.shape[0]
        
        for i in range(num_masks):
            # Process individual mask
            mask = process_single_mask(mask_tensor[i].numpy(), image_array.shape[1], image_array.shape[0])
            
            # Apply color to the mask
            color = colors[i]
            for c in range(3):  # Apply color to each channel
                overlay[:, :, c] += mask * color[c]
                
        # Blend the overlay with the original image
        alpha = settings.MASK_ALPHA
        image_array = cv2.addWeighted(image_array, 1, overlay, alpha, 0)
    
    # Draw bounding boxes and labels
    for box, conf, color in zip(boxes, confidences, colors):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw the dotted bounding box
        draw_dotted_rectangle(image_array, (x1, y1), (x2, y2), color)
        
        # Add label with confidence score
        add_confidence_label(image_array, (x1, y1, x2, y2), conf, color)
    
    return image_array, locule_count

def process_single_mask(mask_data, width, height):
    """Process a single mask."""
    # Binarize mask
    mask = (mask_data > 0.5).astype(np.uint8)
    
    # Resize mask to match the original image dimensions
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Apply Gaussian Blur to soften edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def add_confidence_label(image, box, confidence, color):
    """Add a confidence label to a bounding box."""
    x1, y1, x2, y2 = box
    
    # Format confidence score label
    label = f"Locule {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    
    # Define text position (top-right of bounding box)
    text_x = x2 - text_size[0] - 5
    text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15
    
    # Background rectangle for text
    text_bg_x1, text_bg_y1 = text_x - 2, text_y - text_size[1] - 2
    text_bg_x2, text_bg_y2 = text_x + text_size[0] + 4, text_y + 4
    
    # Draw label background
    cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, thickness=-1)
    
    # Draw label text
    cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

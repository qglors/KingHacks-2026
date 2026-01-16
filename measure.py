import cv2
import numpy as np

def measure_antibody_level(image_path):
    #load image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found."

    #find test
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, white_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    test_contour = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20000: 
            if area > max_area:
                max_area = area
                test_contour = cnt
                
    if test_contour is None:
        return "Error: Could not find the test."

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [test_contour], -1, 255, -1)
    isolated_test_line = cv2.bitwise_and(img, img, mask=mask)
    
    #find ONLY the red line
    hsv = cv2.cvtColor(isolated_test_line, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    target_contour = None
    max_area = 0
    
    for cnt in red_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / h
        
        #distingusih squares/dots (ratio ~1.0) vs lines (ratio > 1.5 or < 0.8)
        is_not_square = (aspect_ratio > 1.5) or (aspect_ratio < 0.8)
        
        if area > 50 and is_not_square: 
            if area > max_area:
                max_area = area
                target_contour = cnt

    if target_contour is None:
        return "Error: No antibody line detected."

    #take antibody level and rescale to calculate percentage
    x, y, w, h = cv2.boundingRect(target_contour)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("final_result.jpg", img)

    roi = gray[y:y+h, x:x+w]
    mean_brightness = np.mean(roi)
    score = 255 - mean_brightness
    
    #calculate percentage
    #assume 120 is the max opacity/brightness
    max_expected = 120.0 
    percentage = (score / max_expected) * 100
    if percentage > 100: percentage = 100

    return f"Raw Score: {round(score, 2)} | Immunity Level: {round(percentage, 1)}%"

#run
print(measure_antibody_level('test_image2.jpeg'))
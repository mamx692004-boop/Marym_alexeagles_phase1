import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None, None

    # Resize for consistency
    img = cv2.resize(img, (800, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5,5), 0)

    # Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Canny edges
    edges = cv2.Canny(blurred, 50, 150)

    # Combine threshold + edges
    combined = cv2.bitwise_or(thresh, edges)

    # Morphological close/open
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return img, cleaned

def find_gear_contour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour, contours

def count_teeth(contour, center, img_shape):
    if contour is None:
        return 0
    
    num_points = 360
    distances = []
    height, width = img_shape
    for angle in range(num_points):
        rad = np.deg2rad(angle)
        max_r = 0
        max_possible_r = int(np.sqrt((width/2)**2 + (height/2)**2)) + 100
        inside = False
        for r in range(10, max_possible_r):
            x = int(center[0] + r*np.cos(rad))
            y = int(center[1] + r*np.sin(rad))
            if x<0 or y<0 or x>=width or y>=height:
                break
            if cv2.pointPolygonTest(contour, (x,y), False) >=0:
                inside = True
            elif inside:
                max_r = r
                break
        distances.append(max_r)
    
    distances = np.array(distances)
    if np.max(distances)>0:
        distances = distances/np.max(distances)
    smoothed = np.convolve(distances, np.ones(10)/10, mode='same')
    peaks, _ = find_peaks(smoothed, distance=10, prominence=0.05)
    return len(peaks)

def highlight_defects(ideal_img, ideal_mask, sample_img, sample_mask):
    # Align centers
    M_ideal = cv2.moments(ideal_mask)
    if M_ideal['m00'] == 0:
        return sample_img.copy(), None, None
    ideal_center = (int(M_ideal['m10']/M_ideal['m00']), int(M_ideal['m01']/M_ideal['m00']))

    M_sample = cv2.moments(sample_mask)
    if M_sample['m00'] == 0:
        return sample_img.copy(), None, None
    sample_center = (int(M_sample['m10']/M_sample['m00']), int(M_sample['m01']/M_sample['m00']))

    dx = ideal_center[0] - sample_center[0]
    dy = ideal_center[1] - sample_center[1]
    M = np.float32([[1,0,dx],[0,1,dy]])
    sample_mask_aligned = cv2.warpAffine(sample_mask, M, (sample_mask.shape[1], sample_mask.shape[0]))

    # Pixel-wise difference
    missing_parts = cv2.bitwise_and(ideal_mask, cv2.bitwise_not(sample_mask_aligned))
    extra_parts   = cv2.bitwise_and(sample_mask_aligned, cv2.bitwise_not(ideal_mask))

    # Result image
    result_img = sample_img.copy()
    # 🔴 خليت كل العيوب (مفقود/زائد) باللون الأحمر
    result_img[(missing_parts>0) | (extra_parts>0)] = [0,0,255]

    # Add legend
    cv2.putText(result_img, "Red: Defects (Missing/Extra)", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    return result_img, missing_parts, extra_parts

def compare_gears(ideal_path, sample_path):
    ideal_img, ideal_bin = load_and_preprocess(ideal_path)
    sample_img, sample_bin = load_and_preprocess(sample_path)
    
    if ideal_img is None or sample_img is None:
        return
    
    ideal_contour, _ = find_gear_contour(ideal_bin)
    sample_contour, _ = find_gear_contour(sample_bin)
    if ideal_contour is None or sample_contour is None:
        print(f"Error: cannot find contours in {sample_path}")
        return
    
    M_ideal = cv2.moments(ideal_contour)
    M_sample = cv2.moments(sample_contour)
    if M_ideal['m00'] == 0 or M_sample['m00'] == 0:
        print(f"Error: invalid contour moments in {sample_path}")
        return
    
    ideal_center = (int(M_ideal['m10']/M_ideal['m00']), int(M_ideal['m01']/M_ideal['m00']))
    sample_center = (int(M_sample['m10']/M_sample['m00']), int(M_sample['m01']/M_sample['m00']))
    
    ideal_teeth = count_teeth(ideal_contour, ideal_center, ideal_img.shape[:2])
    sample_teeth = count_teeth(sample_contour, sample_center, sample_img.shape[:2])

    result_img, missing_parts, extra_parts = highlight_defects(ideal_img, ideal_bin, sample_img, sample_bin)
    
    print(f"Results for {os.path.basename(sample_path)}:")
    print(f"  Ideal teeth: {ideal_teeth}")
    print(f"  Sample teeth: {sample_teeth}")
    if sample_teeth < ideal_teeth:
        print(f"  Missing teeth: {ideal_teeth - sample_teeth}")
    elif sample_teeth > ideal_teeth:
        print(f"  Extra teeth: {sample_teeth - ideal_teeth}")
    else:
        print("  Tooth count OK")
    if missing_parts is not None and extra_parts is not None:
        print(f"  Defect area: {np.sum((missing_parts>0) | (extra_parts>0))} pixels\n")

    # Plot results
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(cv2.cvtColor(ideal_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ideal Gear")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Sample Gear")
    axes[1].axis("off")
    axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Defects Highlighted")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

def main():
    image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not image_files:
        print("No images found")
        return
    
    ideal_file = next((f for f in image_files if 'ideal' in f.lower()), image_files[0])
    sample_files = [f for f in image_files if f != ideal_file]
    
    for sample_file in sample_files:
        compare_gears(ideal_file, sample_file)

if __name__ == "__main__":
    main()








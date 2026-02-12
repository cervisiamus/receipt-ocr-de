import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# --- ENVIRONMENT CONFIGURATION ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["OMP_NUM_THREADS"] = "4"

# Initialize PaddleOCR (Replacing EasyOCR)
# lang='de' is crucial for 'SUMME', 'UID' context if needed, but we check strings manually
ocr_engine = PaddleOCR(use_angle_cls=True, lang='de', use_gpu=False, show_log=False)

def add_black_bottom(image, height_percentage=0.10):
    """
    Adds a black bar to the bottom of the image.
    This creates an artificial border if the receipt is cropped at the bottom.
    """
    h, w = image.shape[:2]
    bar_height = int(h * height_percentage)
    
    # Create a black image with the required width and height
    black_bar = np.zeros((bar_height, w, 3), dtype="uint8")
    
    # Vertically stack the original photo and the black bar
    combined_image = np.vstack((image, black_bar))
    return combined_image

def order_points(pts):
    """Sorts 4 points in the order: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Performs perspective alignment (Perspective Transform)"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width and height of the new flattened receipt
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Coordinates of the ideal rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # OpenCV magic: calculate the transformation matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def get_content_area(image):
    """
    Step 1: Two-Pass OCR (Scout Pass).
    Finds the Y-coordinates of 'UID' (top) and 'SUMME' (bottom) to isolate items.
    """
    # Converting to BGR for Paddle (it prefers 3 channels)
    if len(image.shape) == 2:
        image_ocr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_ocr = image

    # Run PaddleOCR instead of EasyOCR
    results = ocr_engine.ocr(image_ocr, cls=True)
    
    h, w = image.shape[:2]
    
    # Default boundaries
    top_y = 0
    bottom_y = h

    if results and results[0]:
        for line in results[0]:
            # Paddle format: line = [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence) ]
            bbox = line[0]
            text = line[1][0]
            text_upper = text.upper()
            
            # Look for UID (header anchor)
            if "UID" in text_upper or "U1D" in text_upper:
                # bbox[0][1] is the Top-Left Y coordinate
                top_y = int(bbox[0][1])
                print(f"Anchor found: Header (UID) at Y={top_y}")

            # Look for SUMME (footer anchor)
            if "SUMME" in text_upper or "5UMME" in text_upper or "SUMN" in text_upper:
                # bbox[2][1] is the Bottom-Right Y coordinate
                bottom_y = int(bbox[2][1])
                print(f"Anchor found: Footer (SUMME) at Y={bottom_y}")

    # Crop the image with a small margin (padding)
    padding = 10
    start_y = max(0, top_y - padding)
    end_y = min(h, bottom_y + padding)

    return image[start_y:end_y, :]

def split_receipt_vertical(image, ratio=0.60):
    """
    Step 2: Vertical Split.
    Splits the image into two parts: Left (items) and Right (prices).
    """
    h, w = image.shape[:2]
    
    # Calculate the split point on the X-axis
    split_x = int(w * ratio)
    
    # Crop the image
    left_part = image[:, 0:split_x]
    right_part = image[:, split_x:w]
    
    return left_part, right_part

def scan_receipt(image_path):
    """Processes the raw image and returns a cropped, flattened, B&W receipt"""
    # 1. Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return None
        
    # --- NEW STEP: Add artificial bottom border ---
    print("Adding artificial bottom border for better detection...")
    image = add_black_bottom(image)

    orig = image.copy()
    
    # 2. Resize for faster contour detection
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # 3. Convert to grayscale and detect edges (Canny)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # 4. Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # If no receipt is found, use original
    if screenCnt is None:
        print("Warning: Receipt contours not found. Using the original photo.")
        warped = orig
    else:
        # 5. Crop the receipt
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    # 6. Convert to Grayscale
    final_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    return final_image

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    test_image = "test_images/receipt_check_croped.png" 
    
    print("Processing image geometry and basic preprocessing...")
    processed_image = scan_receipt(test_image)
    
    if processed_image is not None:
        # --- STAGE 1: CROP TO CONTENT (ROI Detection) ---
        print("Searching for UID and SUMME anchors...")
        content_image = get_content_area(processed_image)
        
        # --- STAGE 2: VERTICAL SPLIT ---
        print("Splitting image into Left (Items) and Right (Prices)...")
        # Используем ratio=0.75, чтобы дать больше места названиям товаров,
        # так как цены обычно узкие и справа
        left_img, right_img = split_receipt_vertical(content_image, ratio=0.70)
        
        # Save results to disk instead of showing window (better for headless/remote)
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/final_left.png", left_img)
        cv2.imwrite("output/final_right.png", right_img)
        print("Saved split images to 'output/' folder.")

        # --- FINAL OCR ---
        print("Running OCR on the PRICES section (Right Side)...")
        # PaddleOCR call on the right part image
        res_right = ocr_engine.ocr(right_img, cls=True)

        print("\n" + "="*30)
        print("DETECTED PRICES/NUMBERS:")
        print("="*30)
        
        if res_right and res_right[0]:
            for line in res_right[0]:
                text = line[1][0]
                conf = line[1][1]
                # Просто выводим всё, что нашли справа
                print(f"{text} \t(conf: {conf:.2f})")
        else:
            print("No text detected on the right side.")
        print("="*30)
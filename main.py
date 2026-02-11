import cv2
import numpy as np
import easyocr

def add_black_bottom(image, height_percentage=0.10):
    """
    Adds a black bar to the bottom of the image.
    This creates an artificial border if the receipt is cropped at the bottom.
    height_percentage: height of the bar relative to image height (0.10 = 10%)
    """
    h, w = image.shape[:2]
    bar_height = int(h * height_percentage)
    
    # Create a black image with the required width and height
    # (using 3 color channels to match the original image)
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
    # Initialize a fast reader for anchor detection (only English for speed)
    reader_fast = easyocr.Reader(['en'], gpu=True) # Set gpu=False if no GPU
    
    # Detailed pass to get bounding boxes
    results = reader_fast.readtext(image, detail=1)
    
    h, w = image.shape[:2]
    
    # Default boundaries (if anchors are not found)
    top_y = 0
    bottom_y = h

    for (bbox, text, prob) in results:
        text_upper = text.upper()
        
        # Look for UID (header anchor). Using 'in' to catch 'UID:' or 'U1D'
        if "UID" in text_upper or "U1D" in text_upper:
            # bbox[0][1] is the Top-Left Y coordinate
            top_y = int(bbox[0][1])
            print(f"Anchor found: Header (UID) at Y={top_y}")

        # Look for SUMME (footer anchor). Using 'in' for 'SUMME' or '5UMME'
        if "SUMME" in text_upper or "5UMME" in text_upper or "SUMN" in text_upper:
            # bbox[2][1] is the Bottom-Right Y coordinate
            bottom_y = int(bbox[2][1])
            print(f"Anchor found: Footer (SUMME) at Y={bottom_y}")

    # Crop the image with a small margin (padding) to avoid cutting text
    padding = 10
    start_y = max(0, top_y - padding)
    end_y = min(h, bottom_y + padding)

    return image[start_y:end_y, :]

def split_receipt_vertical(image, ratio=0.60):
    """
    Step 2: Vertical Split.
    Splits the image into two parts: Left (items) and Right (prices).
    ratio=0.70 means 70% for the left side and 30% for the right side.
    """
    h, w = image.shape[:2]
    
    # Calculate the split point on the X-axis
    split_x = int(w * ratio)
    
    # Crop the image: [y_start:y_end, x_start:x_end]
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

    # 3. Convert to grayscale and detect edges (Canny) to find the receipt paper
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # 4. Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # Find a contour with exactly 4 corners
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # If no receipt is found on the background, return the original image
    if screenCnt is None:
        print("Warning: Receipt contours not found. Using the original photo.")
        warped = orig
    else:
        # 5. Crop the receipt from the ORIGINAL (large) photo
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    
    # --- NEW STEP: BLACK & WHITE CONVERSION ---
    
    # 6. Convert the final cropped receipt to Grayscale
    final_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    
    return final_image

# --- TESTING BLOCK ---
if __name__ == "__main__":
    test_image = "test_images/receipt_check_croped.png" 
    
    # Step 1: Initial geometric and color processing
    print("Processing image geometry and basic preprocessing...")
    processed_image = scan_receipt(test_image)
    
    if processed_image is not None:
        # --- STAGE 1: CROP TO CONTENT (ROI Detection) ---
        print("Searching for UID and SUMME anchors to isolate the item list...")
        # We pass the processed B&W image to find anchors and get the cropped area
        content_image = get_content_area(processed_image)
        
        # --- STAGE 2: VERTICAL SPLIT ---
        print("Splitting image into Left (Items) and Right (Prices) parts...")
        # Using 75/25 ratio to accommodate long product names
        left_img, right_img = split_receipt_vertical(content_image, ratio=0.7)
        
        # Visual debugging for Split
        # This helps verify that prices are isolated and names are not cut off
        cv2.imshow("Stage 2: Left Side (Items)", left_img)
        cv2.imshow("Stage 2: Right Side (Prices)", right_img)
        
        print("Split complete. Check the windows and press any key to continue to OCR...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Step 2: Initialize Final OCR Reader
        # We'll need German for items, but maybe just English for price numbers
        print("Initializing EasyOCR for final extraction...")
        reader = easyocr.Reader(['de', 'en'], gpu=True)
        
        # --- STAGE 3 PREVIEW (We will refine this next) ---
        print("Ready for independent OCR on each side.")
        # For now, we just print a placeholder message
        print("Next step: OCR with Allowlist for prices and text matching.")
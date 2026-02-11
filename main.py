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
    
    # 6. Convert the final cropped receipt to Grayscale (removes color noise)
    # This is usually the best format for Deep Learning OCRs like EasyOCR
    final_image = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Optional: Strict Black and White (Adaptive Thresholding)
    # If the text is too faint, you can uncomment the line below to force strict B&W.
    # final_image = cv2.adaptiveThreshold(final_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return final_image

# --- TESTING BLOCK ---
if __name__ == "__main__":
    test_image = "test_images/receipt_check_croped.png" 
    
    # Step 1: Process the image with OpenCV
    print("Processing image geometry and color...")
    processed_image = scan_receipt(test_image)
    
    if processed_image is not None:
        # Show the visual B&W result (press any key to close)
        # We expect to see a flat receipt cropped exactly along the added black border
        cv2.imshow("Scanned B&W Receipt", cv2.resize(processed_image, (500, 800)))
        print("Press any key in the image window to continue to OCR...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Step 2: Initialize EasyOCR
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['de', 'en'])
        
        # Step 3: Extract text
        print("Extracting text...")
        text_results = reader.readtext(processed_image, detail=0)
        
        # Step 4: Print the results
        print("\n--- EXTRACTED RECEIPT TEXT ---")
        for line in text_results:
            print(line)
        print("------------------------------\n")
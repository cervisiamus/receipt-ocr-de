import os
import sys
import logging

# --- ENVIRONMENT STABILITY CONFIG ---
# 1. Disable the experimental PIR engine (fixed for 2.6.2)
os.environ["FLAGS_enable_pir_api"] = "0"
# 2. Prevent OpenMP runtime errors on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 3. Disable MKLDNN to ensure stability on Intel U-series CPUs
os.environ["FLAGS_use_mkldnn"] = "0"
# 4. Stop Paddle from checking online sources
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Suppress debug logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

try:
    import paddle
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: Installation is broken. Run 'uv sync --force'.")
    sys.exit(1)

def start_receipt_scan():
    """
    Scans the receipt image using PaddlePaddle 2.6.2 and PaddleOCR 2.7.0.
    Optimized for Intel i5-10210U.
    """
    
    img_path = os.path.join("test_images", "receipt_check_croped.png")
    
    if not os.path.exists(img_path):
        print(f"Error: File not found: {img_path}")
        return

    print(f"Engine Ready: Paddle {paddle.__version__} (CPU Mode)")

    try:
        # Initializing OCR engine. 
        # Models will download once to C:/Users/<User>/.paddleocr/
        ocr = PaddleOCR(lang='en', use_angle_cls=False)
        
        print(f"Scanning: {img_path}")
        result = ocr.ocr(img_path)

        if not result or result[0] is None:
            print("No text found. Try adding a small white border to the image.")
            return

        print("\n" + "="*35)
        print("      RAW OCR OUTPUT")
        print("="*35)
        
        for line in result[0]:
            # result structure: [ [box], (text, confidence) ]
            text = line[1][0]
            confidence = line[1][1]
            
            if confidence > 0.4:
                print(f"Found: {text:<18} | Conf: {confidence:.2f}")

    except Exception as e:
        print(f"Runtime error: {e}")

if __name__ == "__main__":
    start_receipt_scan()
# services/ocr_service.py
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import re
import io
import logging

# Initialize once (EasyOCR downloads models on first run; this might take a moment)
easyocr_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have GPU and want faster results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------- Preprocessing helpers --------------------
def load_image_cv(image_path):
    """Load image as BGR (OpenCV)."""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # fallback to PIL if cv2 can't read path with unicode etc.
        pil = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def deskew(image):
    """Attempt to deskew the image using largest text contour orientation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # binarize for moment calculation
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(bw < 255))
    if coords.shape[0] < 10:
        return image  # nothing to deskew
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    logger.info(f"Deskew angle: {angle:.2f}")
    return rotated


def increase_contrast_and_sharpen(gray):
    """CLAHE + unsharp mask to improve contrast and stroke visibility."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    # Unsharp mask
    blur = cv2.GaussianBlur(cl, (3, 3), 0)
    sharp = cv2.addWeighted(cl, 1.5, blur, -0.5, 0)
    return sharp


def thicken_strokes(binary):
    """Use morphological operations to thicken faint pen strokes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated


def preprocess_image(image_path):
    """
    Full preprocessing pipeline returning both a "tesseract-friendly" PIL image
    and a "easyocr-friendly" OpenCV image (both grayscale/binary).
    """
    img = load_image_cv(image_path)
    img = deskew(img)

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to remove small noise
    gray = cv2.medianBlur(gray, 3)

    # Increase contrast + sharpen
    gray = increase_contrast_and_sharpen(gray)

    # Binarize using Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark
    # count white vs black
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Thicken strokes for faint pen
    binary = thicken_strokes(binary)

    # Return:
    # - PIL image for pytesseract (RGB)
    # - OpenCV grayscale/binary for EasyOCR
    pil_for_tesseract = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
    easyocr_img = binary.copy()

    return pil_for_tesseract, easyocr_img


# -------------------- OCR extraction helpers --------------------
def run_tesseract(pil_image, config=None):
    """
    Return text and optional per-line confidences (if available).
    pil_image: PIL Image object
    config: extra config string for tesseract (e.g. "--psm 6 --oem 3")
    """
    if config is None:
        config = "--oem 3 --psm 6"  # good default: assume a single uniform block
    try:
        # Use image_to_string for plain text
        text = pytesseract.image_to_string(pil_image, config=config)
        # Also get data table to potentially inspect confidences
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=config)
    except Exception as e:
        logger.exception("Tesseract failed")
        return "", None

    # Build simple line list with average confidences
    lines = []
    if data and "line_num" in data:
        current_line_num = -1
        current_words = []
        confs = []
        for i in range(len(data["level"])):
            ln = data.get("line_num", [None])[i]
            txt = data.get("text", [""])[i]
            conf = data.get("conf", ["-1"])[i]
            if ln is None:
                continue
            # if new line starts, flush previous
            if ln != current_line_num and current_line_num != -1:
                joined = " ".join([w for w in current_words if w.strip()])
                avg_conf = np.mean([float(c) for c in confs]) if confs else -1.0
                lines.append((joined, float(avg_conf)))
                current_words = []
                confs = []
            current_line_num = ln
            if txt.strip():
                current_words.append(txt)
                try:
                    confs.append(float(conf))
                except Exception:
                    confs.append(-1.0)
        # flush last
        if current_words:
            joined = " ".join([w for w in current_words if w.strip()])
            avg_conf = np.mean([float(c) for c in confs]) if confs else -1.0
            lines.append((joined, float(avg_conf)))

    # if we failed to parse lines, fallback to whole text
    if not lines:
        flat_text = text.strip()
        if flat_text:
            lines = [(flat_text, -1.0)]
    return text.strip(), lines


def run_easyocr(cv_image):
    """
    Run EasyOCR on the preprocessed binary image.
    Returns paragraph text (joined) and the list of detected strings.
    """
    try:
        # easyocr expects either path or numpy array; passing binary image (uint8)
        results = easyocr_reader.readtext(cv_image, detail=0, paragraph=True)
        # results is list of strings (paragraph True merges into paragraphs)
        joined = "\n".join(results).strip()
        return joined, results
    except Exception:
        logger.exception("EasyOCR failed")
        return "", []


# -------------------- Cleaning & merging --------------------
def clean_ocr_text(text: str) -> str:
    """
    Basic post-processing to remove garbage, fix ligatures, collapse whitespace, etc.
    """
    if not text:
        return ""

    # unify line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # common OCR ligature/garbage fixes
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "\u200b": "",  # zero-width space
        "\x0c": "",    # form feed from tesseract sometimes
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove very weird characters, keep printable unicode and basic punctuation.
    # Allow letters, numbers, common punctuation and newlines.
    text = re.sub(r"[^\w\s\.\,\:\;\?\!\-\(\)\/\'\"\%\—\–\u00C0-\u017F]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse multiple newlines to single
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing whitespace on each line
    lines = [ln.strip() for ln in text.splitlines()]
    # Remove empty lines at start/end
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    cleaned = "\n".join(lines)
    # final whitespace collapse
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def merge_texts(tess_lines, easy_lines, raw_tess_text, raw_easy_text):
    """
    Merge Tesseract and EasyOCR outputs intelligently.
    - tess_lines: list of (line_text, avg_conf) tuples from tesseract
    - easy_lines: list of strings from easyocr
    - raw_tess_text, raw_easy_text: full raw text fallbacks
    Strategy:
      - Prefer high-confidence tesseract lines (conf >= 50)
      - Use easyocr lines where tesseract confidence is low or absent
      - Avoid duplicate lines using normalization
    """
    output_lines = []
    seen_norm = set()

    def normalize(s):
        s2 = s.lower()
        s2 = re.sub(r"[^\w]", "", s2)
        return s2

    # Add tesseract high-confidence first
    if tess_lines:
        for line, conf in tess_lines:
            if not line or line.strip() == "":
                continue
            n = normalize(line)
            if n in seen_norm:
                continue
            if conf >= 50 or conf == -1:  # -1 means unknown; allow it but may be low quality
                output_lines.append(line.strip())
                seen_norm.add(n)

    # Now add easyocr lines if they are new
    if easy_lines:
        for e in easy_lines:
            if not e or e.strip() == "":
                continue
            n = normalize(e)
            if n in seen_norm:
                continue
            output_lines.append(e.strip())
            seen_norm.add(n)

    # If nothing collected, fallback to the best raw text we have
    if not output_lines:
        fallback = raw_tess_text or raw_easy_text or ""
        if fallback:
            # Split fallback sensibly into lines
            flines = [ln.strip() for ln in fallback.splitlines() if ln.strip()]
            for ln in flines:
                n = normalize(ln)
                if n not in seen_norm:
                    output_lines.append(ln)
                    seen_norm.add(n)

    final = "\n".join(output_lines).strip()
    return final


# -------------------- Public API --------------------
def extract_text_from_image(image_path: str) -> str:
    """
    Full pipeline to extract (typed + handwritten) text from image_path.
    Returns cleaned text ready to put in a DOCX.
    """
    # 1) Preprocess to get two variants
    pil_for_tesseract, easyocr_img = preprocess_image(image_path)

    # 2) Tesseract extraction
    raw_tess_text, tess_lines = run_tesseract(pil_for_tesseract, config="--oem 3 --psm 6")
    logger.info(f"Tesseract raw length: {len(raw_tess_text)}; lines: {len(tess_lines) if tess_lines else 0}")

    # 3) EasyOCR extraction (handwriting)
    raw_easy_text, easy_lines = run_easyocr(easyocr_img)
    logger.info(f"EasyOCR detected {len(easy_lines)} items; raw len: {len(raw_easy_text)}")

    # 4) Merge intelligently
    merged = merge_texts(tess_lines or [], easy_lines or [], raw_tess_text, raw_easy_text)

    # 5) Clean and return
    cleaned = clean_ocr_text(merged)
    logger.info(f"Final cleaned text length: {len(cleaned)}")
    return cleaned


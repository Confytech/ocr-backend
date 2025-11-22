# services/ocr_service.py
import cv2
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import re
import logging
from typing import List, Tuple, Dict

# -------------------- Config --------------------
TESS_CONF_THRESHOLD = 50
ALPHA_RATIO_THRESHOLD = 0.35
MAX_GIBBERISH_TOKEN_RATIO = 0.6

# initialize EasyOCR (downloads models on first run)
easyocr_reader = easyocr.Reader(['en'], gpu=False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -------------------- Helpers --------------------
def _is_likely_gibberish(line: str) -> bool:
    if not line or not line.strip():
        return True
    s = line.strip()
    letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", s)
    alpha_ratio = len(letters) / max(1, len(s))
    tokens = [t for t in re.split(r"\s+", s) if t]
    if len(tokens) == 0:
        return True
    gibberish_tokens = 0
    for t in tokens:
        alphacount = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", t))
        if alphacount < 2:
            gibberish_tokens += 1
    token_gib_ratio = gibberish_tokens / len(tokens)
    if alpha_ratio < ALPHA_RATIO_THRESHOLD and token_gib_ratio > MAX_GIBBERISH_TOKEN_RATIO:
        return True
    if all(len(t) == 1 for t in tokens) and len(tokens) > 4:
        return True
    return False


def _clean_line(line: str) -> str:
    if not line:
        return ""
    line = line.replace("\x0c", " ").replace("\u200b", "")
    line = re.sub(r"\s+", " ", line).strip()
    return line


# -------------------- Image loading & preprocessing --------------------
def _load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _preprocess_for_tesseract(img_bgr: np.ndarray) -> Image.Image:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)
    den = cv2.fastNlMeansDenoising(cl, None, h=10, templateWindowSize=7, searchWindowSize=21)
    blur = cv2.GaussianBlur(den, (3, 3), 0)
    sharp = cv2.addWeighted(den, 1.4, blur, -0.4, 0)
    th = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    pil = Image.fromarray(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB))
    return pil


def _preprocess_for_handwriting(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    den = cv2.normalize(den, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    den = clahe.apply(den)
    return den


# -------------------- OCR Runners --------------------
def _run_tesseract_on_pil(pil_img: Image.Image, config: str = "--oem 3 --psm 6") -> Tuple[str, List[Tuple[str, float]]]:
    raw = pytesseract.image_to_string(pil_img, config=config)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=config)
    lines: List[Tuple[str, float]] = []
    if data and "line_num" in data:
        current_line = []
        confs = []
        current_ln = -1
        for i in range(len(data["level"])):
            ln = data["line_num"][i]
            txt = data["text"][i]
            conf = data["conf"][i]
            if ln != current_ln and current_ln != -1:
                joined = " ".join([w for w in current_line if w.strip()])
                try:
                    avg_conf = float(np.mean([float(c) for c in confs])) if confs else -1.0
                except Exception:
                    avg_conf = -1.0
                lines.append((joined, avg_conf))
                current_line = []
                confs = []
            current_ln = ln
            if txt and txt.strip():
                current_line.append(txt)
                try:
                    confs.append(float(conf))
                except Exception:
                    confs.append(-1.0)
        if current_line:
            joined = " ".join([w for w in current_line if w.strip()])
            try:
                avg_conf = float(np.mean([float(c) for c in confs])) if confs else -1.0
            except Exception:
                avg_conf = -1.0
            lines.append((joined, avg_conf))
    if not lines:
        if raw and raw.strip():
            lines = [(raw.strip(), -1.0)]
    return raw.strip(), lines


def _run_easyocr_on_img_detailed(img_np: np.ndarray) -> Tuple[str, List[Tuple[str, float]]]:
    try:
        raw_res = easyocr_reader.readtext(img_np, detail=1, paragraph=False)
    except Exception:
        logger.exception("EasyOCR error")
        return "", []
    lines: List[Tuple[str, float]] = []
    for item in raw_res:
        if len(item) >= 3:
            txt = (item[1] or "").strip()
            conf = float(item[2]) if item[2] is not None else -1.0
            lines.append((txt, conf))
    joined = "\n".join([t for t, _ in lines])
    return joined, lines


# -------------------- Merge & finalize --------------------
def _merge_results(tess_lines: List[Tuple[str, float]], easy_lines: List[Tuple[str, float]],
                   raw_tess: str, raw_easy: str) -> str:
    final_lines: List[str] = []
    seen_norm = set()

    def norm(s):
        return re.sub(r"[^\w]", "", s.lower())

    # Accept easyocr lines first (filtered)
    for e_text, e_conf in easy_lines or []:
        line = _clean_line(e_text)
        if not line:
            continue
        if _is_likely_gibberish(line):
            continue
        n = norm(line)
        if n in seen_norm:
            continue
        final_lines.append(line)
        seen_norm.add(n)

    # Accept tesseract only if high confidence or clearly typed-like
    for t_text, t_conf in tess_lines or []:
        line = _clean_line(t_text)
        if not line:
            continue
        if t_conf != -1 and t_conf < TESS_CONF_THRESHOLD:
            if _is_likely_gibberish(line):
                continue
        else:
            if _is_likely_gibberish(line):
                continue
        n = norm(line)
        if n in seen_norm:
            continue
        final_lines.append(line)
        seen_norm.add(n)

    # fallback to raw texts if nothing
    if not final_lines:
        fallback = ((raw_easy or "") + "\n" + (raw_tess or "")).strip()
        for ln in [l.strip() for l in fallback.splitlines() if l.strip()]:
            if not _is_likely_gibberish(ln):
                final_lines.append(_clean_line(ln))

    return "\n".join(final_lines)


def _final_cleanup(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x0c", " ").replace("\u200b", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l.strip() for l in text.splitlines()]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    lines = [re.sub(r"[ \t]{2,}", " ", l) for l in lines]
    return "\n".join(lines)


# -------------------- Public API --------------------
def extract_lines_with_conf(image_path: str) -> List[Dict]:
    """
    Returns list of dicts: {"text":..., "source":"tesseract"|"easyocr", "conf":float}
    """
    img_bgr = _load_image_bgr(image_path)

    pil_for_tess = _preprocess_for_tesseract(img_bgr)
    raw_tess, tess_lines = _run_tesseract_on_pil(pil_for_tess)

    easy_img = _preprocess_for_handwriting(img_bgr)
    raw_easy, easy_lines = _run_easyocr_on_img_detailed(easy_img)

    items = []
    for t_line, t_conf in tess_lines or []:
        line = _clean_line(t_line)
        if not line:
            continue
        items.append({"text": line, "source": "tesseract", "conf": float(t_conf)})

    for e_line, e_conf in easy_lines or []:
        line = _clean_line(e_line)
        if not line:
            continue
        items.append({"text": line, "source": "easyocr", "conf": float(e_conf)})

    return items


def extract_text_from_image(image_path: str) -> str:
    img_bgr = _load_image_bgr(image_path)

    pil_for_tess = _preprocess_for_tesseract(img_bgr)
    raw_tess, tess_lines = _run_tesseract_on_pil(pil_for_tess)

    easy_img = _preprocess_for_handwriting(img_bgr)
    raw_easy, easy_lines = _run_easyocr_on_img_detailed(easy_img)

    merged = _merge_results(tess_lines or [], easy_lines or [], raw_tess, raw_easy)
    cleaned = _final_cleanup(merged)
    logger.info(f"Final cleaned text length: {len(cleaned)}")
    return cleaned


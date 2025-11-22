# app.py
from flask import Flask, request, send_file, jsonify, send_from_directory, render_template, after_this_request
from services.ocr_service import extract_text_from_image, extract_lines_with_conf
from services.docx_service import create_docx
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024  # 12 MB

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.ico")

@app.route("/", methods=["GET"])
def test():
    return jsonify({"message": "OCR Backend Running"})

@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")

# Existing upload -> returns docx (keeps behavior)
@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error":"No image provided"}), 400
    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error":"Empty filename"}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error":"Unsupported file type"}), 400

    original = secure_filename(image_file.filename)
    unique = f"{uuid.uuid4().hex}_{original}"
    path = os.path.join(UPLOAD_FOLDER, unique)
    image_file.save(path)

    try:
        extracted_text = extract_text_from_image(path)
        docx_path = create_docx(extracted_text, original)
        @after_this_request
        def cleanup(response):
            try:
                os.remove(path)
                # optionally remove docx after send? keep for now
            except Exception:
                pass
            return response
        return send_file(docx_path, as_attachment=True, download_name=f"{os.path.splitext(original)[0]}.docx")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New: return structured lines for editing
@app.route("/api/extract", methods=["POST"])
def api_extract():
    if "image" not in request.files:
        return jsonify({"error":"No image provided"}), 400
    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error":"Empty filename"}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error":"Unsupported file type"}), 400

    original = secure_filename(image_file.filename)
    tmpname = f"tmp_{uuid.uuid4().hex}_{original}"
    tmp_path = os.path.join(UPLOAD_FOLDER, tmpname)
    image_file.save(tmp_path)

    try:
        lines = extract_lines_with_conf(tmp_path)
        return jsonify({"lines": lines})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# Accept final text and return DOCX
@app.route("/api/generate_docx", methods=["POST"])
def api_generate_docx():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error":"No text provided"}), 400
    text = data["text"]
    filename = data.get("filename", f"extracted_{uuid.uuid4().hex}")
    # Use create_docx(text, original_filename)
    docx_path = create_docx(text, filename)
    return send_file(docx_path, as_attachment=True, download_name=f"{os.path.splitext(filename)[0]}.docx")

if __name__ == "__main__":
    app.run(debug=True)


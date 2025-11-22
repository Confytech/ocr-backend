from flask import Flask, request, send_file, jsonify, send_from_directory, render_template, after_this_request
from services.ocr_service import extract_text_from_image  # This must combine typed + handwritten OCR
from services.docx_service import create_docx
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)

# ---------- Config ----------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}
MAX_FILE_SIZE_MB = 10

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024  # 10 MB

# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(filename):
    ext = os.path.splitext(filename)[1]
    return f"{uuid.uuid4().hex}{ext}"

# ---------- Routes ----------
@app.route("/favicon.ico")
def favicon():
    return send_from_directory("static", "favicon.ico")

@app.route("/", methods=["GET"])
def test():
    return jsonify({"message": "OCR Backend Running"})

@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")

@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Generate a safe, unique filename
    original_filename = secure_filename(image_file.filename)
    unique_filename = generate_unique_filename(original_filename)
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    image_file.save(image_path)

    # Process OCR (typed + handwritten)
    extracted_text = extract_text_from_image(image_path)

    # Create DOCX
    docx_path = create_docx(extracted_text, unique_filename)
    base_name = os.path.splitext(original_filename)[0]

    # ---------- Cleanup after sending ----------
    @after_this_request
    def remove_files(response):
        try:
            os.remove(image_path)
            os.remove(docx_path)
        except Exception as e:
            print(f"Error cleaning files: {e}")
        return response

    return send_file(
        docx_path,
        as_attachment=True,
        download_name=f"{base_name}.docx"
    )

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)

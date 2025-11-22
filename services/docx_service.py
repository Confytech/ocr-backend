from docx import Document
import os

OUTPUT_FOLDER = "output"

def create_docx(text, original_filename):
    doc = Document()
    doc.add_heading("Extracted Text", level=1)
    doc.add_paragraph(text)

    docx_filename = original_filename.split('.')[0] + ".docx"
    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)

    doc.save(docx_path)
    return docx_path


"""
export_and_render_pdf.py
========================
Exports poster.pptx to poster.pdf using Microsoft PowerPoint COM,
and then renders poster.pdf to poster_preview.png using PyMuPDF (fitz)
for automated visual layout auditing.
"""

import os
import sys
from pathlib import Path
import win32com.client
import pymupdf  # PyMuPDF

def export_pptx_to_pdf(pptx_abs_path: str, pdf_abs_path: str):
    print(f"Exporting {pptx_abs_path} to PDF...")
    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    # ppSaveAsPDF = 32
    try:
        deck = powerpoint.Presentations.Open(pptx_abs_path, WithWindow=False)
        deck.SaveAs(pdf_abs_path, 32)
        deck.Close()
        print(f"Exported successfully to {pdf_abs_path}")
    finally:
        powerpoint.Quit()

def render_pdf_to_png(pdf_abs_path: str, png_abs_path: str, dpi: int = 150):
    print(f"Rendering {pdf_abs_path} to {png_abs_path} at {dpi} DPI...")
    doc = pymupdf.open(pdf_abs_path)
    page = doc.load_page(0)
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(png_abs_path)
    print(f"Rendered preview image: {png_abs_path} ({pix.width}x{pix.height})")

def main():
    base_dir = Path("poster").resolve()
    pptx_path = base_dir / "poster.pptx"
    pdf_path = base_dir / "poster.pdf"
    png_path = base_dir / "poster_preview.png"

    if not pptx_path.exists():
        pptx_path = Path("In_context_learning/poster/poster.pptx").resolve()
        pdf_path = Path("In_context_learning/poster/poster.pdf").resolve()
        png_path = Path("In_context_learning/poster/poster_preview.png").resolve()

    export_pptx_to_pdf(str(pptx_path), str(pdf_path))
    render_pdf_to_png(str(pdf_path), str(png_path), dpi=150)

    # Also sync to In_context_learning/poster
    other_pdf = Path("In_context_learning/poster/poster.pdf").resolve()
    other_png = Path("In_context_learning/poster/poster_preview.png").resolve()
    if pdf_path != other_pdf:
        import shutil
        shutil.copy2(pdf_path, other_pdf)
        shutil.copy2(png_path, other_png)
        print(f"Copied PDF and preview PNG to {other_pdf.parent}")

if __name__ == "__main__":
    main()

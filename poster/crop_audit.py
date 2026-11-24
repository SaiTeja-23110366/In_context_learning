from PIL import Image
from pathlib import Path

def crop_sections():
    img_path = Path("poster/poster_preview.png")
    if not img_path.exists():
        img_path = Path("In_context_learning/poster/poster_preview.png")
    
    img = Image.open(img_path)
    w, h = img.size
    print(f"Full poster image size: {w} x {h}")
    
    crops_dir = Path("poster/crops")
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Header (top 16% of poster)
    header = img.crop((0, 0, w, int(h * 0.16)))
    header.save(crops_dir / "header_crop.png")
    
    # 2. Column 1 (left 34%, middle 80%)
    col1 = img.crop((0, int(h * 0.14), int(w * 0.35), int(h * 0.95)))
    col1.save(crops_dir / "col1_crop.png")
    
    # 3. Column 2 (middle 34%, middle 80%)
    col2 = img.crop((int(w * 0.33), int(h * 0.14), int(w * 0.68), int(h * 0.95)))
    col2.save(crops_dir / "col2_crop.png")
    
    # 4. Column 3 (right 35%, middle 80%)
    col3 = img.crop((int(w * 0.65), int(h * 0.14), w, int(h * 0.95)))
    col3.save(crops_dir / "col3_crop.png")
    
    # 5. Footer (bottom 7% of poster)
    footer = img.crop((0, int(h * 0.93), w, h))
    footer.save(crops_dir / "footer_crop.png")
    
    print("Crops created in poster/crops/!")

if __name__ == "__main__":
    crop_sections()

from PIL import Image

# 1. Open the TGA file
input_path = "./data/processed/inD/map/03_heckstrasse.tga"
try:
    img = Image.open(input_path)
except FileNotFoundError:
    print(f"Error: '{input_path}' not found.")
    exit(1)
except Exception as e:
    print(f"Error opening '{input_path}': {e}")
    exit(1)

# 2. Rotate 180 degrees
#    You can use either rotate(180) or transpose(ROTATE_180). Both work the same for 180°.
rotated = img.rotate(180)

# 3. Save as PNG
output_path = "./data/processed/inD/map/03_heckstrasse.png"
try:
    rotated.save(output_path, format="PNG")
    print(f"Saved rotated image to '{output_path}'.")
except Exception as e:
    print(f"Error saving '{output_path}': {e}")

"""
Convert Grooten book HTML to Obsidian markdown with Chesser diagrams.

Takes an HTML file from grooten/text/ and creates a markdown file with:
- Text content preserved
- Chess diagrams converted to Chesser format (interactive in Obsidian)

Usage:
    python html_to_chesser.py grooten/text/part0006.html

Output:
    Creates a markdown file in the project root with the same name.
"""

from __future__ import annotations

import sys
import re
import os
from pathlib import Path
from html.parser import HTMLParser
from PIL import Image
import numpy as np

# Add ChessVisionAI to path for the model
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "ChessVisionAI"))

import tensorflow as tf
from config import class_labels, class_to_fen, img_width, img_height


def load_model():
    """Load the trained ChessVisionAI model."""
    model_path = SCRIPT_DIR / "ChessVisionAI" / "models" / "chess_classifier_10k.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(str(model_path))


def image_to_fen(image_path: Path, model) -> str:
    """Convert a chess diagram image to FEN notation."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Crop border if not square
    if abs(w - h) > 5:
        # Try to crop to square
        size = min(w, h)
        left = (w - size) // 2
        top = (h - size) // 2
        img = img.crop((left, top, left + size, top + size))
    
    # Also try cropping 2px border
    w, h = img.size
    if w > 20 and h > 20:
        img_test = img.crop((2, 2, w-2, h-2))
        # Make square
        w2, h2 = img_test.size
        size = min(w2, h2)
        left = (w2 - size) // 2
        top = (h2 - size) // 2
        img = img_test.crop((left, top, left + size, top + size))
    
    w, h = img.size
    sq_w = w // 8
    sq_h = h // 8
    
    # Extract squares
    squares = []
    for row in range(8):
        for col in range(8):
            x0, x1 = col * sq_w, (col + 1) * sq_w
            y0, y1 = row * sq_h, (row + 1) * sq_h
            sq = img.crop((x0, y0, x1, y1))
            squares.append(sq)
    
    # Batch classify
    batch = np.array([
        np.array(sq.resize((img_width, img_height))) / 255.0
        for sq in squares
    ], dtype=np.float32)
    
    predictions = model.predict(batch, verbose=0)
    
    # Classify with confidence thresholding
    empty_idx = class_labels.index("empty")
    classifications = []
    
    for p in predictions:
        idx = np.argmax(p)
        confidence = p[idx]
        label = class_labels[idx]
        empty_conf = p[empty_idx]
        
        # Pawn thresholding
        if label in ("w_pawn", "b_pawn"):
            if confidence < 0.95 or empty_conf > 0.02:
                label = "empty"
        elif label != "empty" and confidence < 0.50:
            if empty_conf > 0.15:
                label = "empty"
        
        classifications.append(label)
    
    # Build FEN
    fen_rows = []
    for row in range(8):
        fen_row = ""
        empty_count = 0
        for col in range(8):
            label = classifications[row * 8 + col]
            fen_piece = class_to_fen[label]
            if fen_piece == "1":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += fen_piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row or "8")
    
    return "/".join(fen_rows)


class ChessHTMLParser(HTMLParser):
    """Parse HTML and convert to markdown with Chesser diagrams."""
    
    def __init__(self, images_dir: Path, model):
        super().__init__()
        self.images_dir = images_dir
        self.model = model
        self.output = []
        self.current_text = ""
        self.in_bold = False
        self.in_italic = False
        self.in_list = False
        self.list_counter = 0
        self.skip_content = False  # Skip head, script, etc.
        self.current_tag = ""
        self.pending_newlines = 0
        
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        self.current_tag = tag
        
        if tag in ("head", "script", "style", "link", "meta"):
            self.skip_content = True
            return
        
        if tag == "img":
            src = attrs_dict.get("src", "")
            alt = attrs_dict.get("alt", "")
            
            # Check if it's a chess diagram (in images folder)
            if "images/" in src and src.endswith(".jpeg"):
                self._flush_text()
                image_name = src.split("/")[-1]
                image_path = self.images_dir / image_name
                
                if image_path.exists():
                    try:
                        fen = image_to_fen(image_path, self.model)
                        # Chesser format for Obsidian
                        self.output.append(f"\n```chesser\nfen: {fen}\n```\n")
                    except Exception as e:
                        self.output.append(f"\n[Chess diagram: {image_name} - Error: {e}]\n")
                else:
                    # Probably a photo, not a diagram
                    if "images/000" in src:
                        self.output.append(f"\n[Image: {image_name}]\n")
            return
        
        if tag == "b":
            self.in_bold = True
        elif tag == "i":
            self.in_italic = True
        elif tag == "ol":
            self.in_list = True
            self.list_counter = 0
        elif tag == "li":
            self._flush_text()
            self.list_counter += 1
            value = attrs_dict.get("value")
            if value:
                self.list_counter = int(value)
            self.current_text = f"{self.list_counter}. "
        elif tag == "p":
            self._flush_text()
            class_name = attrs_dict.get("class", "")
            if "chap" in class_name:
                self.pending_newlines = 2
            elif "head" in class_name:
                self.pending_newlines = 2
            elif "level" in class_name:
                self.pending_newlines = 2
            else:
                self.pending_newlines = 1
        elif tag == "div":
            self._flush_text()
    
    def handle_endtag(self, tag):
        if tag in ("head", "script", "style"):
            self.skip_content = False
            return
        
        if tag == "b":
            self.in_bold = False
        elif tag == "i":
            self.in_italic = False
        elif tag == "ol":
            self.in_list = False
        elif tag in ("p", "div", "li"):
            self._flush_text()
            self.output.append("\n")
    
    def handle_data(self, data):
        if self.skip_content:
            return
        
        text = data
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        if not text.strip():
            return
        
        # Apply formatting
        if self.in_bold:
            text = f"**{text.strip()}**"
        if self.in_italic:
            text = f"*{text.strip()}*"
        
        self.current_text += text
    
    def _flush_text(self):
        if self.current_text.strip():
            # Add pending newlines
            prefix = "\n" * self.pending_newlines
            self.output.append(prefix + self.current_text.strip())
            self.pending_newlines = 0
        self.current_text = ""
    
    def get_markdown(self) -> str:
        self._flush_text()
        result = "".join(self.output)
        # Clean up multiple newlines
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()


def convert_html_to_markdown(html_path: Path, output_path: Path, model) -> None:
    """Convert an HTML file to markdown with Chesser diagrams."""
    
    # Find images directory
    images_dir = html_path.parent.parent / "images"
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Parse and convert
    parser = ChessHTMLParser(images_dir, model)
    parser.feed(html_content)
    markdown = parser.get_markdown()
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    
    print(f"Created: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python html_to_chesser.py grooten/text/part0006.html")
        print("       python html_to_chesser.py grooten/text/part0006.html output.md")
        sys.exit(1)
    
    html_path = Path(sys.argv[1])
    if not html_path.exists():
        print(f"Error: File not found: {html_path}")
        sys.exit(1)
    
    # Output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = SCRIPT_DIR / f"{html_path.stem}.md"
    
    print("Loading model...")
    model = load_model()
    
    print(f"Converting: {html_path}")
    convert_html_to_markdown(html_path, output_path, model)


if __name__ == "__main__":
    main()

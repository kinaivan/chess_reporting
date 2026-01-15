"""
Simple chessboard diagram to FEN converter for static diagrams.

This script is a first step toward recognizing positions from chess
book diagrams like the one provided. It assumes:

* The input image is a cropped 8x8 board (no extra margins).
* White is at the bottom, Black at the top.
* All diagrams of a given source use the same piece graphics.
* You provide template images for each piece type in a folder.

Usage (from the project root):

    python image_to_fen.py /path/to/board.jpg /path/to/templates

Where the templates directory contains 12 PNG files named:

    wP.png, wN.png, wB.png, wR.png, wQ.png, wK.png
    bP.png, bN.png, bB.png, bR.png, bQ.png, bK.png

Each template should be a small, tightly cropped image of the piece
from the same kind of diagram (same font/style).

This script:
1. Splits the board image into an 8x8 grid.
2. For each square, runs simple template matching to decide whether a
   piece is present and which one.
3. Outputs a FEN string for the position.

This is intentionally simple and tuned for static, consistent diagrams.
For more robustness (varying styles, photos of boards, etc.) a more
advanced CV / ML pipeline would be needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np

PIECE_FILES: Dict[str, str] = {
    "P": "wP.png",
    "N": "wN.png",
    "B": "wB.png",
    "R": "wR.png",
    "Q": "wQ.png",
    "K": "wK.png",
    "p": "bP.png",
    "n": "bN.png",
    "b": "bB.png",
    "r": "bR.png",
    "q": "bQ.png",
    "k": "K.png",
}


@dataclass
class Template:
    piece: str  # FEN letter, e.g. "P", "k"
    image: np.ndarray  # grayscale template
    edges: np.ndarray  # edge-detected template


def load_templates(templates_dir: Path) -> Dict[str, Template]:
    """
    Load piece templates from the provided directory.

    Expects files named as in PIECE_FILES.
    """
    templates: Dict[str, Template] = {}
    for piece, filename in PIECE_FILES.items():
        path = templates_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Template for piece '{piece}' not found at {path}. "
                f"Expected file {filename} in {templates_dir}"
            )
        img_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise RuntimeError(f"Failed to read template image: {path}")

        # Normalize and compute edges to make matching less sensitive to
        # light/dark square backgrounds.
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        img_edges = cv2.Canny(img_blur, 50, 150)

        templates[piece] = Template(piece=piece, image=img_gray, edges=img_edges)
    return templates


def classify_square(
    roi: np.ndarray,
    templates: Dict[str, Template],
    threshold: float = 0.5,
) -> Optional[str]:
    """
    Classify a single square ROI as a piece or empty.

    Uses normalized cross-correlation template matching. Returns a FEN
    piece letter (e.g. "P", "k") or None if no template matches well.
    """
    # Normalize ROI size a bit (optional small border to avoid edges)
    h, w = roi.shape[:2]
    margin_y = max(1, h // 16)
    margin_x = max(1, w // 16)
    roi_cropped = roi[margin_y : h - margin_y, margin_x : w - margin_x]

    roi_gray = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2GRAY) if roi_cropped.ndim == 3 else roi_cropped

    # Work in edge space to reduce the impact of square color and
    # lighting; this helps especially for light (white) pieces.
    roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    roi_edges = cv2.Canny(roi_blur, 50, 150)

    best_piece: Optional[str] = None
    best_score: float = -1.0

    for piece, tmpl in templates.items():
        tmpl_edges = tmpl.edges
        th, tw = tmpl_edges.shape[:2]

        # If the template is larger than the ROI (e.g. piece images are
        # very zoomed in), downscale it to fit inside the ROI while
        # keeping aspect ratio.
        scale = min(roi_edges.shape[0] / th, roi_edges.shape[1] / tw)
        if scale < 1.0:
            new_size = (max(1, int(tw * scale)), max(1, int(th * scale)))
            tmpl_edges = cv2.resize(tmpl_edges, new_size, interpolation=cv2.INTER_AREA)

        # If the (possibly resized) template is still too big in one
        # dimension, skip it for this ROI.
        th2, tw2 = tmpl_edges.shape[:2]
        if th2 > roi_edges.shape[0] or tw2 > roi_edges.shape[1]:
            continue

        res = cv2.matchTemplate(roi_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_piece = piece

    # Apply the presence threshold after we've found the best match.
    if best_score >= threshold:
        return best_piece
    return None


def image_to_board(
    image_path: Path,
    templates: Dict[str, Template],
    threshold: float = 0.5,
) -> list[list[Optional[str]]]:
    """
    Convert a board image into a 2D array of piece codes or None.

    Assumes the image is a tightly cropped 8x8 board, white at bottom.
    Returns a list of 8 ranks (from top to bottom), each a list of 8
    squares (from left to right).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]
    square_h = h // 8
    square_w = w // 8

    board: list[list[Optional[str]]] = []

    for rank_idx in range(8):  # 0 = top rank (Black side)
        rank: list[Optional[str]] = []
        y0 = rank_idx * square_h
        y1 = (rank_idx + 1) * square_h if rank_idx < 7 else h
        for file_idx in range(8):  # 0 = a-file (left)
            x0 = file_idx * square_w
            x1 = (file_idx + 1) * square_w if file_idx < 7 else w
            roi = img[y0:y1, x0:x1]
            piece = classify_square(roi, templates, threshold=threshold)
            rank.append(piece)
        board.append(rank)

    return board


def board_to_fen(board: list[list[Optional[str]]]) -> str:
    """
    Convert a board array (top rank first) to a FEN piece placement
    string (no side-to-move / castling info).
    """
    fen_ranks = []
    for rank in board:
        empty_run = 0
        parts = []
        for sq in rank:
            if sq is None:
                empty_run += 1
            else:
                if empty_run > 0:
                    parts.append(str(empty_run))
                    empty_run = 0
                parts.append(sq)
        if empty_run > 0:
            parts.append(str(empty_run))
        fen_ranks.append("".join(parts) or "8")
    return "/".join(fen_ranks)


def main(argv: list[str]) -> None:
    if len(argv) < 3:
        print(
            "Usage: python image_to_fen.py /path/to/board.jpg /path/to/templates [threshold]\n"
            "Example: python image_to_fen.py diagram.jpg templates 0.55"
        )
        raise SystemExit(1)

    image_path = Path(argv[1])
    templates_dir = Path(argv[2])
    threshold = float(argv[3]) if len(argv) >= 4 else 0.55

    templates = load_templates(templates_dir)
    board = image_to_board(image_path, templates, threshold=threshold)
    fen = board_to_fen(board)

    # Print just the piece placement part; user can add side-to-move etc.
    print(fen)


if __name__ == "__main__":
    main(sys.argv)


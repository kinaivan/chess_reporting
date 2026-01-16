"""
ChessVisionAI-based image to FEN converter.

Uses a trained neural network to classify chess pieces.
Requires a trained model in ChessVisionAI/models/

Usage:
    python chessvision_to_fen.py path/to/board.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Add ChessVisionAI to path
CHESSVISION_DIR = Path(__file__).parent / "ChessVisionAI"
sys.path.insert(0, str(CHESSVISION_DIR))

from config import class_labels, class_to_fen, img_width, img_height


def load_model(model_path: Path | None = None) -> tf.keras.Model:
    """Load the trained ChessVisionAI model."""
    if model_path is None:
        model_path = CHESSVISION_DIR / "models" / "chess_classifier_10k.keras"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run datagen.ipynb and train.ipynb first."
        )
    
    return tf.keras.models.load_model(str(model_path))


def classify_chessboard(
    image_path: str | Path,
    model: tf.keras.Model,
) -> list[list[str]]:
    """
    Classify a chessboard image into a 2D array of piece labels.
    
    Args:
        image_path: Path to a square chessboard image
        model: Trained Keras model
        
    Returns:
        8x8 list of piece labels (e.g., 'w_pawn', 'b_rook', 'empty')
    """
    chessboard = Image.open(image_path).convert("RGB")
    
    # Assume square board
    board_width, board_height = chessboard.size
    square_w = board_width // 8
    square_h = board_height // 8
    
    squares = []
    for row in range(8):
        for col in range(8):
            left = col * square_w
            upper = row * square_h
            right = left + square_w
            lower = upper + square_h
            square = chessboard.crop((left, upper, right, lower))
            squares.append(square)
    
    # Preprocess and batch classify
    batch = np.array([
        np.array(sq.resize((img_width, img_height))) / 255.0
        for sq in squares
    ], dtype=np.float32)
    
    predictions = model.predict(batch, verbose=0)
    
    # Classify with confidence thresholding
    # Pawns are often confused with empty squares, so require very high confidence
    empty_idx = class_labels.index("empty")
    
    classifications = []
    for p in predictions:
        idx = np.argmax(p)
        confidence = p[idx]
        label = class_labels[idx]
        empty_conf = p[empty_idx]
        
        # For pawns: require 95%+ confidence, OR if empty has ANY probability, reject
        if label in ("w_pawn", "b_pawn"):
            if confidence < 0.95 or empty_conf > 0.02:
                label = "empty"
        # For other pieces with low confidence, check empty
        elif label != "empty" and confidence < 0.50:
            if empty_conf > 0.15:
                label = "empty"
        
        classifications.append(label)
    
    # Reshape into 8x8 board
    board = [
        [classifications[row * 8 + col] for col in range(8)]
        for row in range(8)
    ]
    return board


def board_to_fen(board: list[list[str]]) -> str:
    """
    Convert board configuration to FEN notation.
    
    Args:
        board: 8x8 list of piece labels from classify_chessboard()
        
    Returns:
        FEN piece placement string
    """
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for square in row:
            fen_piece = class_to_fen[square]
            if fen_piece == "1":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += fen_piece
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows)


def image_to_fen(image_path: str | Path, model: tf.keras.Model) -> str:
    """
    Convert a chessboard image to FEN notation.
    
    Args:
        image_path: Path to a square chessboard image
        model: Trained Keras model
        
    Returns:
        FEN piece placement string
    """
    board = classify_chessboard(image_path, model)
    return board_to_fen(board)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python chessvision_to_fen.py path/to/board.png")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print("Loading model...")
    model = load_model()
    
    print(f"Processing: {image_path}")
    fen = image_to_fen(image_path, model)
    print(f"FEN: {fen}")


if __name__ == "__main__":
    main()


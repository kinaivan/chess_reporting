import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img  # type: ignore
from PIL import Image
import numpy as np
from config import class_to_fen, img_width, img_height


# Visualize the board configuration
def visualize_board(board_configuration):
    """
    Visualize the board configuration using a table plot.

    Parameters:
    board_configuration: list of list of str
        A 2D list representing the board configuration where each element
        is a string indicating the type of piece on the square.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    table = plt.table(cellText=board_configuration, cellLoc="center", loc="center")
    table.scale(1, 2)
    ax.axis("off")
    plt.show()


# Function to convert board configuration to FEN
def board_to_fen(board_configuration):
    """
    Convert the board configuration to FEN notation.

    Parameters:
    board_configuration: list of list of str
        A 2D list representing the board configuration where each element
        is a string indicating the type of piece on the square.

    Returns:
    str
        The FEN notation string representing the board configuration.
    """
    fen_rows = []
    for row in board_configuration:
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
    fen = "/".join(fen_rows)
    return fen


def preprocess_image(
    image, is_quantized_model=False, target_size=(img_width, img_height)
):
    """
    Preprocess the image for model inference.

    Parameters:
    image: str or PIL.Image.Image
        If a string, this should be the path to the image file.
        If a PIL.Image.Image object, it uses the provided image directly.
    is_quantized_model: bool
        If True, process the image for a quantized model.
        If False, process the image for a non-quantized model.
    target_size: tuple of int
        The target size to resize the image to (width, height).

    Returns:
    np.ndarray
        The preprocessed image as a numpy array.
    """
    if isinstance(image, str):
        img = load_img(image, target_size=target_size)
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
        img = img.resize(target_size)
    else:
        raise ValueError(
            "The image parameter should be either a file path or a PIL.Image.Image object."
        )

    if not is_quantized_model:
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize for non-quantized model
    else:
        img_array = np.array(img, dtype=np.uint8)  # Use uint8 for quantized model

    assert len(img_array.shape) == 3, f"Expected 3D image tensor, got {img_array.shape}"

    # Add batch dimension if necessary
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

"""
Generate training dataset for ChessVisionAI.

This script creates augmented images from the base pieces and empty board.
Run this before train_model.py.

Usage:
    python generate_dataset.py
"""

from PIL import Image
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    array_to_img
)
import numpy as np
import random
import uuid
import os
from concurrent.futures import ThreadPoolExecutor

# Config
base_dir = "data/pieces"
empty_board = "data/empty_board.png"
backgrounds = ["dataset/squares/square_0_1.png", "dataset/squares/square_0_2.png"]
training_data_dir = "dataset/training"
test_data_dir = "dataset/test"
squares_dir = "dataset/squares"
num_images = 2_000  # Images to generate per piece (reduced for memory)

# Create directories
os.makedirs(training_data_dir, exist_ok=True)
os.makedirs(test_data_dir, exist_ok=True)
os.makedirs(squares_dir, exist_ok=True)


def create_empty_squares():
    """Create empty squares from empty board."""
    print("Creating empty squares from board...")
    board_image = Image.open(empty_board)

    board_width, board_height = board_image.size
    if board_width != board_height:
        raise ValueError("The board image is not square!")

    square_size = board_width // 8

    count = 0
    for row in range(8):
        for col in range(8):
            left = col * square_size
            top = row * square_size
            right = left + square_size
            bottom = top + square_size

            square_image = board_image.crop((left, top, right, bottom))
            square_name = f"square_{row}_{col}.png"
            square_image.save(os.path.join(squares_dir, square_name))
            count += 1

    print(f"Empty squares generated: {count}")


# Define augmentation generators - GENTLE augmentation to keep pieces complete
datagen = ImageDataGenerator(
    rotation_range=5,           # Very slight rotation only
    width_shift_range=0.05,     # Minimal shift
    height_shift_range=0.05,    # Minimal shift
    shear_range=0.02,           # Almost no shear
    zoom_range=0.1,             # Slight zoom variation
    horizontal_flip=False,      # Don't flip pieces!
    vertical_flip=False,        # Don't flip pieces!
    brightness_range=[0.8, 1.2], # Slight brightness variation
    fill_mode="nearest",
)

datagen_empty = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.02,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)


def add_background(piece_path, background_paths, num_images_per_bg):
    """Overlay image on background."""
    piece = Image.open(piece_path).convert("RGBA")
    images = []
    # Limit to avoid memory issues
    images_per_bg = min(num_images_per_bg, 100)
    for bg_path in background_paths:
        background = Image.open(bg_path).convert("RGBA")
        background = background.resize(piece.size, Image.Resampling.LANCZOS)
        for _ in range(images_per_bg):
            combined = Image.alpha_composite(background, piece)
            images.append(combined)
    return images


def augment_and_save(images, output_path, num_augmented=1000):
    """Generate and save augmented images for the pieces."""
    os.makedirs(output_path, exist_ok=True)
    augmentations_per_image = max(1, num_augmented // len(images)) if images else 0
    
    for img in images:
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=None, save_format="png"):
            augmented_img = batch[0]
            img_out = array_to_img(augmented_img, scale=True)
            unique_id = uuid.uuid4().hex
            filename = f"augmented_{unique_id}.png"
            img_out.save(os.path.join(output_path, filename))
            i += 1
            if i >= augmentations_per_image:
                break
        
        # Clear memory
        del x


def is_valid_image(img):
    """Check if an image is valid and non-empty."""
    if img is None:
        return False
    if img.size == (0, 0):
        return False
    if np.array(img).mean() == 0:
        return False
    return True


def augment_empty_square(img, num_augmented):
    """Augment and save empty squares separately."""
    augmented_images = []
    x = img_to_array(img)
    if x.size == 0:
        print(f"Skipping invalid image with shape: {img.size}")
        return []
    x = x.reshape((1,) + x.shape)

    for batch in datagen_empty.flow(x, batch_size=1, save_to_dir=None, save_format="png"):
        augmented_img = batch[0]
        if np.array(augmented_img).mean() == 0:
            continue
        augmented_images.append(augmented_img)
        if len(augmented_images) >= num_augmented:
            break

    return augmented_images


def load_and_classify_empty_squares(empty_square_dir):
    """Load and classify empty squares by type."""
    unique_squares = {
        "left_hedge": [],
        "bottom_hedge": [],
        "dark_square": None,
        "light_square": None,
    }

    for filename in os.listdir(empty_square_dir):
        piece_path = os.path.join(empty_square_dir, filename)
        try:
            empty_image = Image.open(piece_path).convert("RGBA")
            if not is_valid_image(empty_image):
                continue
        except Exception as e:
            print(f"Error loading image {piece_path}: {e}")
            continue

        _, row, col = filename.rstrip(".png").split("_")
        row, col = int(row), int(col)

        if col == 0:
            unique_squares["left_hedge"].append(empty_image)
        elif row == 7:
            unique_squares["bottom_hedge"].append(empty_image)
        elif unique_squares["dark_square"] is None and np.array(empty_image).mean() < 127:
            unique_squares["dark_square"] = empty_image
        elif unique_squares["light_square"] is None and np.array(empty_image).mean() >= 127:
            unique_squares["light_square"] = empty_image

    return (
        unique_squares["left_hedge"]
        + unique_squares["bottom_hedge"]
        + [unique_squares["dark_square"], unique_squares["light_square"]]
    )


def process_empty_squares(unique_images, output_train_dir, output_test_dir, num_augmented):
    """Process and save empty square augmentations."""
    all_augmented_images = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(augment_empty_square, img, num_augmented)
            for img in unique_images
        ]
        for future in futures:
            all_augmented_images.extend(future.result())

    random.shuffle(all_augmented_images)
    split_index = int(len(all_augmented_images) * 0.8)
    train_images = all_augmented_images[:split_index]
    test_images = all_augmented_images[split_index:]

    os.makedirs(output_train_dir, exist_ok=True)
    for img in train_images:
        img_out = array_to_img(img, scale=True)
        unique_id = uuid.uuid4().hex
        filename = f"augmented_{unique_id}.png"
        img_out.save(os.path.join(output_train_dir, filename))

    os.makedirs(output_test_dir, exist_ok=True)
    for img in test_images:
        img_out = array_to_img(img, scale=True)
        unique_id = uuid.uuid4().hex
        filename = f"augmented_{unique_id}.png"
        img_out.save(os.path.join(output_test_dir, filename))


def process_piece(color, filename, backgrounds, num_images, training_data_dir, test_data_dir):
    """Process a single piece type."""
    piece_type = filename.split("_")[1].split(".")[0]
    piece_path = os.path.join(base_dir, color, filename)
    
    print(f"  Processing {color} {piece_type}...")
    images = add_background(piece_path, backgrounds, num_images)

    random.shuffle(images)
    split_index = int(len(images) * 0.8)
    train_images = images[:split_index]
    test_images = images[split_index:]

    output_piece_train_dir = os.path.join(training_data_dir, f"{color[0]}_{piece_type}")
    os.makedirs(output_piece_train_dir, exist_ok=True)
    augment_and_save(train_images, output_piece_train_dir)

    output_piece_test_dir = os.path.join(test_data_dir, f"{color[0]}_{piece_type}")
    os.makedirs(output_piece_test_dir, exist_ok=True)
    augment_and_save(test_images, output_piece_test_dir)


def main():
    print("=" * 50)
    print("ChessVisionAI Dataset Generator")
    print("=" * 50)
    
    # Step 1: Create empty squares
    create_empty_squares()
    
    # Step 2: Generate piece dataset (sequential to avoid memory issues)
    print("\nGenerating piece variations...")
    for color in ["black", "white"]:
        color_dir = os.path.join(base_dir, color)
        for filename in os.listdir(color_dir):
            process_piece(
                color,
                filename,
                backgrounds,
                num_images,
                training_data_dir,
                test_data_dir,
            )

    print("Done generating pieces dataset!")

    # Step 3: Generate empty squares dataset
    print("\nGenerating empty square variations...")
    unique_images = load_and_classify_empty_squares(squares_dir)
    unique_images = [img for img in unique_images if img is not None]

    process_empty_squares(
        unique_images,
        os.path.join(training_data_dir, "empty"),
        os.path.join(test_data_dir, "empty"),
        round(num_images / 15),
    )

    print("Done generating empty squares dataset!")
    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print(f"Training data: {training_data_dir}")
    print(f"Test data: {test_data_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

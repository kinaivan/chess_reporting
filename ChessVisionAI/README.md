# Chess Vision AI

Chess Vision AI is a project aimed at extracting the state of a chessboard from a screenshot using a trained AI model. This project is part of my training on AI, and it achieves an accuracy of 99.8% in classifying chessboard squares. As long as the chessboard is the default one on [chess.com](https://chess.com). For now.

The model is trained using transfer learning from MobileNetV2 and can classify the state of a chessboard with high accuracy.

![demo](https://i.imgur.com/1VhFZub.png)

## Usage

### Generate a dataset

Run the [`datagen.ipynb`](https://github.com/Coriou/ChessVisionAI/blob/main/datagen.ipynb) notebook to generate a dataset from the pieces & board in the `data` directory. It took my Intel CPU 20 minutes to generate roughly 10k images / piece.

The data is the default board & pieces found on [chess.com](https://chess.com).

### Train the model

Run the [`train.ipynb`](https://github.com/Coriou/ChessVisionAI/blob/main/train.ipynb) notebook to train the model on the generated dataset. The last cell will evaluate the model's performances on the test dataset.

Took just over an hour to train on my Intel CPU thanks to transfer learning from MobileNetV2 (might be better performing pretrained ImageNet models out there).

### Test

Run the [`predict_board.ipynb`](https://github.com/Coriou/ChessVisionAI/blob/main/predict_board.ipynb) to test it against a screenshot.

Can also test indivual squares and _debug_ the model in the [`infer.ipynb`](https://github.com/Coriou/ChessVisionAI/blob/main/infer.ipynb).

## Detect chessboard from a larger image

This project enhances the original model to detect and extract a chessboard from any screenshot, even if the image contains other elements or the chessboard is rotated. This solution leverages and improves upon an existing project called [tensorflow_chessbot](https://github.com/Elucidation/tensorflow_chessbot), which implemented [chessboard squares detection](https://github.com/Elucidation/tensorflow_chessbot/blob/master/tensorflow_compvision.ipynb).

### Key Enhancements:

- **Rotation Detection and correction**: Integrated a rotation detection mechanism to ensure the chessboard is correctly aligned before detection, mitigating issues with skewed images.
- **Code Modernization**: Updated the code to work with the latest versions of TensorFlow (2.x) and SciPy.
- **Improved Image Quality**: Applied bicubic interpolation during image rotation to reduce aliasing and improve the visual quality of the rotated images.
- **Code Cleanup**: Refactored and cleaned up the code for better readability and maintainability.

### Usage:

To use the enhanced chessboard detection, run the [`detect_chessboard.ipynb`](https://github.com/Coriou/ChessVisionAI/blob/main/detect_chessboard.ipynb) notebook. This notebook demonstrates how to detect and extract a chessboard from a larger image, handling various orientations and image sizes.

This project aims to provide a more robust and user-friendly solution for chessboard detection in real-world scenarios.

## Next

In the future, I plan to train the model on more boards and pieces. Currently, the model performs poorly on chessboards from [Lichess](https://lichess.org). Extending the training data to include various board styles should improve performance across different platforms.

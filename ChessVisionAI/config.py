# Class labels as per your training data
class_labels = [
    "b_bishop",
    "b_king",
    "b_knight",
    "b_pawn",
    "b_queen",
    "b_rook",
    "empty",
    "w_bishop",
    "w_king",
    "w_knight",
    "w_pawn",
    "w_queen",
    "w_rook",
]
num_classes = len(class_labels)  # 13: 6 pieces x 2 colors + 1 empty

# Mapping class labels to FEN notation
class_to_fen = {
    "b_bishop": "b",
    "b_king": "k",
    "b_knight": "n",
    "b_pawn": "p",
    "b_queen": "q",
    "b_rook": "r",
    "empty": "1",
    "w_bishop": "B",
    "w_king": "K",
    "w_knight": "N",
    "w_pawn": "P",
    "w_queen": "Q",
    "w_rook": "R",
}

# Input's image size (a chessboard square)
img_width, img_height = 224, 224

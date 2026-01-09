import requests
from typing import List, Dict

USERNAME = "kinaivan"
TIME_CONTROL = "900+10"  # 15|10 in chess.com notation
COMMENTARY_PATH = "/Users/isladonj/Documents/obsidian/obididian-main/Chess/Reports.md"


def fetch_archives(username: str) -> List[str]:
    """Return list of monthly archive URLs for a chess.com user."""
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("archives", [])


def fetch_15_10_games(username: str) -> List[Dict]:
    """
    Fetch all games for the user and filter to 15|10 time control.

    Returns a list of game dicts from chess.com API.
    """
    archives = fetch_archives(username)
    games_15_10 = []

    for archive_url in archives:
        try:
            resp = requests.get(archive_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Failed to fetch archive {archive_url}: {e}")
            continue

        for game in data.get("games", []):
            if game.get("time_control") == TIME_CONTROL:
                games_15_10.append(game)

    return games_15_10


def load_commentary_lines(path: str) -> List[str]:
    """
    Load commentary from the given file.

    Assumes each non-empty line is commentary for one game, in the same
    order as the games are printed. If there are fewer lines than games,
    remaining games will have empty commentary.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Commentary file not found at: {path}")
        lines = []
    return lines


def normalize_result(raw_result: str) -> str:
    """
    Map chess.com result codes to a simpler 'win/loss/draw/other' label.
    """
    if raw_result == "win":
        return "win"
    if raw_result in {"agreed", "stalemate", "repetition", "timevsinsufficient",
                      "insufficient", "50move", "draw"}:
        return "draw"
    if raw_result in {"checkmated", "resigned", "timeout", "abandoned"}:
        return "loss"
    return raw_result or "unknown"


def print_games_with_commentary(username: str) -> None:
    games = fetch_15_10_games(username)
    commentaries = load_commentary_lines(COMMENTARY_PATH)

    if not games:
        print(f"No 15|10 games found for user '{username}'.")
        return

    print(f"Found {len(games)} 15|10 games for '{username}'.\n")

    for idx, game in enumerate(games):
        white = game.get("white", {})
        black = game.get("black", {})

        white_name = white.get("username", "").lower()
        black_name = black.get("username", "").lower()

        if white_name == username.lower():
            player_color = "white"
            opponent = black
        elif black_name == username.lower():
            player_color = "black"
            opponent = white
        else:
            # Skip games where the user is not one of the players (shouldn't happen)
            continue

        player = white if player_color == "white" else black

        opponent_name = opponent.get("username", "Unknown")
        player_rating = player.get("rating", "N/A")
        opponent_rating = opponent.get("rating", "N/A")

        raw_result = player.get("result", "")
        result_simple = normalize_result(raw_result)

        commentary = commentaries[idx] if idx < len(commentaries) else ""

        print(f"Game {idx + 1}:")
        print(f"  Opponent: {opponent_name}")
        print(f"  Ratings: {username} ({player_rating}) vs {opponent_name} ({opponent_rating})")
        print(f"  Result for {username}: {result_simple} (raw: {raw_result})")
        if commentary:
            print(f"  Commentary: {commentary}")
        else:
            print("  Commentary: (none)")
        print()  # blank line between games


if __name__ == "__main__":
    print_games_with_commentary(USERNAME)
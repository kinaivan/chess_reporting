import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests
from typing import List, Dict, Tuple

from engine_commentary import generate_engine_commentary_for_game

USERNAME = "kinaivan"
TIME_CONTROL = "900+10"  # 15|10 in chess.com notation
COMMENTARY_PATH = "/Users/isladonj/Documents/obsidian/obididian-main/Chess/Reports.md"
GAMES_JSON_PATH = os.path.join(os.path.dirname(__file__), "games.json")
FULL_REPORT_PATH = "/Users/isladonj/Documents/obsidian/obididian-main/Chess/Full_report.md"

# Only consider games from 2026 onward when fetching from the API.
MIN_2026_END_TIME = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())


def fetch_archives(username: str) -> List[str]:
    """Return list of monthly archive URLs for a chess.com user."""
    # I did 'curl https://api.chess.com/pub/player/kinaivan/games/2026/01 | jq '.games[] | select(.time_control == "900+10")' > games.json' to get the games.json file
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
            if game.get("time_control") != TIME_CONTROL:
                continue
            end_time = game.get("end_time", 0)
            # Skip games that finished before 2026.
            if not isinstance(end_time, int) or end_time < MIN_2026_END_TIME:
                continue
            games_15_10.append(game)

    return games_15_10


def load_games_from_file(path: str = GAMES_JSON_PATH) -> List[Dict]:
    """
    Load games from a local JSON file instead of querying the API.

    The file is expected to contain a list of game dicts in the same
    format as returned by the chess.com API.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Games file not found at: {path}")
        return []

    if not isinstance(data, list):
        print(f"Games file at {path} is not a list; got {type(data)} instead.")
        return []

    # Just in case the file has mixed time controls, filter again.
    return [g for g in data if g.get("time_control") == TIME_CONTROL]


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


def get_game_date(game: Dict) -> str:
    """
    Derive a human-readable game date (YYYY-MM-DD).

    Prefer the 'end_time' UNIX timestamp from the JSON. If that is
    missing, fall back to UTCDate/Date tags in the PGN.
    """
    end_time = game.get("end_time")
    if isinstance(end_time, int) and end_time > 0:
        try:
            dt = datetime.fromtimestamp(end_time, tz=timezone.utc)
            return dt.date().isoformat()
        except Exception:
            pass

    pgn_text = str(game.get("pgn", ""))
    utc_date = ""
    date_tag = ""
    for line in pgn_text.splitlines():
        line = line.strip()
        if line.startswith("[UTCDate "):
            parts = line.split('"')
            if len(parts) >= 2:
                utc_date = parts[1].strip()
        if line.startswith("[Date "):
            parts = line.split('"')
            if len(parts) >= 2:
                date_tag = parts[1].strip()

    date_str = utc_date or date_tag
    if date_str and date_str != "????.??.??":
        # PGN dates are usually in YYYY.MM.DD
        return date_str.replace(".", "-")

    return "Unknown"


def get_opening_name(game: Dict) -> str:
    """
    Derive a human-readable opening name for a game.

    Prefer the 'eco' URL field from the JSON (which points to the
    chess.com opening page). If that is missing, fall back to parsing
    ECOUrl or Opening tags from the PGN.
    """
    eco_url = str(game.get("eco") or "").strip()
    if eco_url:
        slug = eco_url.rstrip("/").split("/")[-1]
        if slug:
            return slug.replace("-", " ")

    pgn_text = str(game.get("pgn", ""))
    for line in pgn_text.splitlines():
        line = line.strip()
        if line.startswith("[Opening "):
            parts = line.split('"')
            if len(parts) >= 2:
                name = parts[1].strip()
                if name:
                    return name
        if line.startswith("[ECOUrl "):
            parts = line.split('"')
            if len(parts) >= 2:
                url = parts[1].strip()
                slug = url.rstrip("/").split("/")[-1]
                if slug:
                    return slug.replace("-", " ")

    return "Unknown"


def fetch_15_10_games_since(username: str, since_end_time: int) -> List[Dict]:
    """
    Fetch 15|10 games from the API that finished after since_end_time.

    This is intended for updating games.json with only new games.
    """
    archives = fetch_archives(username)
    new_games: List[Dict] = []

    for archive_url in archives:
        try:
            resp = requests.get(archive_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Failed to fetch archive {archive_url}: {e}")
            continue

        for game in data.get("games", []):
            if game.get("time_control") != TIME_CONTROL:
                continue
            end_time = game.get("end_time", 0)
            if not isinstance(end_time, int):
                continue
            # Only consider games from 2026 onward and newer than the last
            # recorded end_time we already have.
            if end_time < MIN_2026_END_TIME or end_time <= since_end_time:
                continue
            new_games.append(game)

    # Keep games in chronological order by end_time
    new_games.sort(key=lambda g: g.get("end_time", 0))
    return new_games


def update_games_json(username: str, json_path: str = GAMES_JSON_PATH) -> Tuple[int, int]:
    """
    Read games from json_path, fetch any newer 15|10 games from the API,
    append them, and write back to json_path.

    Returns a tuple: (number of new games appended, number of games that
    were already present before the update).
    """
    existing_games = load_games_from_file(json_path)
    previous_count = len(existing_games)
    if existing_games:
        latest_end_time = max(
            (g.get("end_time", 0) for g in existing_games if isinstance(g.get("end_time", 0), int)),
            default=0,
        )
    else:
        latest_end_time = 0

    print(f"Latest recorded end_time in {json_path}: {latest_end_time}")

    new_games = fetch_15_10_games_since(username, latest_end_time)
    if not new_games:
        print("No new 15|10 games found to append.")
        return 0, previous_count

    updated_games = existing_games + new_games

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated_games, f, indent=2)

    print(f"Appended {len(new_games)} new games to {json_path}. Total games: {len(updated_games)}.")
    return len(new_games), previous_count


def build_markdown_report(
    username: str,
    start_at_game_index: int = 0,
    include_header: bool = True,
) -> str:
    """
    Build a markdown report of games with commentary.

    If start_at_game_index > 0, only games from that index onward are
    printed, but commentary indexing is still based on the full list of
    games so that existing commentary for older games is left unchanged
    and new commentary lines can be appended for newly fetched games.
    """
    # Read games locally from games.json instead of querying the API.
    games = load_games_from_file(GAMES_JSON_PATH)
    commentaries = load_commentary_lines(COMMENTARY_PATH)

    if not games:
        return f"No 15|10 games found for user '{username}'.\n"

    lines: List[str] = []
    if include_header:
        lines.append(f"# Chess report for {username}")

        if start_at_game_index <= 0:
            lines.append("")
            lines.append(f"Total 15|10 games: {len(games)}")
        else:
            lines.append("")
            lines.append(
                f"Total 15|10 games: {len(games)} "
                f"(showing newly fetched games from #{start_at_game_index + 1} onward)"
            )

    for idx, game in enumerate(games[start_at_game_index:], start=start_at_game_index):
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

        game_date = get_game_date(game)

        # Try to get a direct URL to the game on chess.com.
        game_url = game.get("url", "")
        if not game_url:
            # Fallback: try to parse the Link tag from the PGN if present.
            pgn_text = game.get("pgn", "")
            for line in pgn_text.splitlines():
                line = line.strip()
                if line.startswith("[Link "):
                    # Format: [Link "https://..."]
                    parts = line.split('"')
                    if len(parts) >= 2:
                        game_url = parts[1]
                    break

        opening_name = get_opening_name(game)

        # Prefer manually written commentary lines if present. If there is
        # no existing commentary for this game, try to generate a short
        # explanation using the engine instead.
        commentary = commentaries[idx] if idx < len(commentaries) else ""
        if not commentary:
            commentary = generate_engine_commentary_for_game(game, username)

        lines.append("")
        lines.append(f"## Game {idx + 1}")
        lines.append(f"- **Opponent**: {opponent_name}")
        lines.append(f"- **Ratings**: {username} ({player_rating}) vs {opponent_name} ({opponent_rating})")
        lines.append(f"- **Result for {username}**: {result_simple}")
        lines.append(f"- **Date**: {game_date}")
        lines.append(f"- **Opening**: {opening_name}")
        if game_url:
            lines.append(f"- **Game link**: [{game_url}]({game_url})")
        if commentary:
            lines.append(f"- **Commentary**: {commentary}")
        else:
            lines.append(f"- **Commentary**: (none)")

    if include_header:
        lines.append("")
    return "\n".join(lines)


def print_games_with_commentary(username: str, start_at_game_index: int = 0) -> None:
    """
    Convenience wrapper to print the markdown report to stdout.
    """
    report = build_markdown_report(username, start_at_game_index=start_at_game_index)
    print(report)


def count_existing_reported_games(report_path: str = FULL_REPORT_PATH) -> int:
    """
    Count how many games are already present in the full report markdown
    file, by counting headings of the form '## Game X'.
    """
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0

    count = 0
    for line in lines:
        if line.strip().startswith("## Game "):
            count += 1
    return count


def build_summary_report(username: str, top_n_openings: int = 5) -> str:
    """
    Build a markdown summary section with overall results, by color, and
    by the most commonly played openings.
    """
    games = load_games_from_file(GAMES_JSON_PATH)
    if not games:
        return ""

    username_lower = username.lower()

    total_counts: Counter = Counter()
    color_counts: Dict[str, Counter] = {
        "white": Counter(),
        "black": Counter(),
    }
    opening_game_counts: Counter = Counter()
    # opening -> color -> Counter(results)
    opening_color_counts: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: {"white": Counter(), "black": Counter()}
    )

    for game in games:
        white = game.get("white", {}) or {}
        black = game.get("black", {}) or {}

        white_name = str(white.get("username", "")).lower()
        black_name = str(black.get("username", "")).lower()

        if white_name == username_lower:
            color = "white"
            raw_result = str(white.get("result", ""))
        elif black_name == username_lower:
            color = "black"
            raw_result = str(black.get("result", ""))
        else:
            continue

        result = normalize_result(raw_result)
        if result not in {"win", "draw", "loss"}:
            continue

        opening = get_opening_name(game)

        total_counts[result] += 1
        color_counts[color][result] += 1
        opening_game_counts[opening] += 1
        opening_color_counts[opening][color][result] += 1

    total_games = sum(total_counts.values())
    if total_games == 0:
        return ""

    def pct(count: int, total: int) -> float:
        return (count / total * 100.0) if total > 0 else 0.0

    lines: List[str] = []
    lines.append("## Summary and statistics")
    lines.append("")

    # Overall results
    wins = total_counts["win"]
    draws = total_counts["draw"]
    losses = total_counts["loss"]

    lines.append("### Overall results")
    lines.append(f"- **Games**: {total_games}")
    lines.append(
        f"- **Wins**: {wins} ({pct(wins, total_games):.1f}%), "
        f"**Draws**: {draws} ({pct(draws, total_games):.1f}%), "
        f"**Losses**: {losses} ({pct(losses, total_games):.1f}%)"
    )
    lines.append("")

    # By color
    lines.append("### Results by color")
    for color in ["white", "black"]:
        c_counts = color_counts[color]
        c_total = sum(c_counts.values())
        if c_total == 0:
            continue
        cw = c_counts["win"]
        cd = c_counts["draw"]
        cl = c_counts["loss"]
        color_title = color.capitalize()
        lines.append(f"- **{color_title}**: {c_total} games — "
                     f"{cw} wins ({pct(cw, c_total):.1f}%), "
                     f"{cd} draws ({pct(cd, c_total):.1f}%), "
                     f"{cl} losses ({pct(cl, c_total):.1f}%)")
    lines.append("")

    # By opening (top N)
    lines.append(f"### Results by opening (top {top_n_openings} by games played)")
    if not opening_game_counts:
        lines.append("- **No openings to report yet.**")
    else:
        for opening, games_for_opening in opening_game_counts.most_common(top_n_openings):
            lines.append(f"- **{opening}** ({games_for_opening} games)")
            oc_white = opening_color_counts[opening]["white"]
            oc_black = opening_color_counts[opening]["black"]

            w_total = sum(oc_white.values())
            b_total = sum(oc_black.values())

            if w_total > 0:
                w_wins = oc_white["win"]
                w_draws = oc_white["draw"]
                w_losses = oc_white["loss"]
                lines.append(
                    f"  - as White: {w_total} games — "
                    f"{w_wins} wins ({pct(w_wins, w_total):.1f}%), "
                    f"{w_draws} draws ({pct(w_draws, w_total):.1f}%), "
                    f"{w_losses} losses ({pct(w_losses, w_total):.1f}%)"
                )
            if b_total > 0:
                b_wins = oc_black["win"]
                b_draws = oc_black["draw"]
                b_losses = oc_black["loss"]
                lines.append(
                    f"  - as Black: {b_total} games — "
                    f"{b_wins} wins ({pct(b_wins, b_total):.1f}%), "
                    f"{b_draws} draws ({pct(b_draws, b_total):.1f}%), "
                    f"{b_losses} losses ({pct(b_losses, b_total):.1f}%)"
                )
    lines.append("")

    return "\n".join(lines)


def append_new_games_to_full_report(username: str) -> None:
    """
    Ensure that Full_report.md contains a section for every game in
    games.json, without duplicating games that are already present.

    The file is treated as append-only: we never rewrite existing
    content, only append sections for games that are not yet there.
    """
    games = load_games_from_file(GAMES_JSON_PATH)
    if not games:
        print("No games found in games.json; nothing to add to full report.")
        return

    already_reported = count_existing_reported_games(FULL_REPORT_PATH)

    if already_reported >= len(games):
        print("Full_report.md is already up to date with all games.")
        return

    # Decide whether we need to include the header in the new chunk.
    file_exists = os.path.exists(FULL_REPORT_PATH)
    file_is_empty = False
    if file_exists:
        file_is_empty = os.path.getsize(FULL_REPORT_PATH) == 0

    include_header = not file_exists or file_is_empty

    report_chunk = build_markdown_report(
        username,
        start_at_game_index=already_reported,
        include_header=include_header,
    )

    # Build a fresh summary for all games so the last section of the
    # report always reflects current statistics.
    summary_chunk = build_summary_report(username)

    # Append or create the report file as appropriate.
    mode = "a" if file_exists and not file_is_empty else "w"
    with open(FULL_REPORT_PATH, mode, encoding="utf-8") as f:
        if mode == "a":
            # Ensure there's at least one blank line before new content.
            f.write("\n")
        f.write(report_chunk)
        if summary_chunk:
            f.write("\n")
            f.write(summary_chunk)

    newly_added = len(games) - already_reported
    print(
        f"Appended {newly_added} new game(s) to {FULL_REPORT_PATH}. "
        f"Total games in report should now be {len(games)}."
    )


def copy_and_push_full_report() -> None:
    """
    Copy the full report markdown file into the current repository
    folder and push it to GitHub using credentials from the environment.
    """
    repo_dir = os.path.dirname(__file__)
    local_copy_path = os.path.join(repo_dir, "Full_report.md")

    # Copy the report into the current folder
    try:
        shutil.copyfile(FULL_REPORT_PATH, local_copy_path)
    except FileNotFoundError:
        print(f"Full report not found at {FULL_REPORT_PATH}; skipping copy and push.")
        return

    print(f"Copied full report to local repo path: {local_copy_path}")

    token = os.environ.get("GITHUB_PASSWORD")
    if not token:
        print("Environment variable GITHUB_PASSWORD not set; skipping git push.")
        return

    github_username = "kinaivan"
    remote_url = (
        f"https://{github_username}:{token}@github.com/"
        f"{github_username}/chess_reporting.git"
    )

    def run_git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
        )

    # Stage the local copy
    add_result = run_git("add", "Full_report.md")
    if add_result.returncode != 0:
        print(f"git add failed: {add_result.stderr.strip()}")
        return

    # Commit changes (if any)
    commit_result = run_git("commit", "-m", "Update Full_report.md")
    if commit_result.returncode != 0:
        stderr = commit_result.stderr.lower()
        if "nothing to commit" in stderr:
            print("No changes to commit for Full_report.md; skipping push.")
            return
        print(f"git commit failed: {commit_result.stderr.strip()}")
        return

    # Push to GitHub
    push_result = run_git("push", remote_url, "HEAD:main")
    if push_result.returncode != 0:
        print(f"git push failed: {push_result.stderr.strip()}")
        return

    print("Successfully pushed Full_report.md to GitHub.")


if __name__ == "__main__":
    import sys

    fetch_flag = False

    # Simple CLI parsing: expect optional argument of the form fetch=True/False
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower().startswith("fetch="):
            value = arg.split("=", 1)[1].strip().lower()
            if value in {"true", "1", "yes", "y", "t"}:
                fetch_flag = True
            elif value in {"false", "0", "no", "n", "f"}:
                fetch_flag = False
            else:
                print(f"Unrecognized fetch value '{value}', defaulting to fetch=False.")

    if fetch_flag:
        # Update games.json with any new games from the API.
        new_count, previous_count = update_games_json(USERNAME, GAMES_JSON_PATH)
        if new_count == 0:
            print("No new games fetched from the API.")

    # In all cases, make sure Full_report.md has entries for every game
    # in games.json, without duplicating existing game sections.
    append_new_games_to_full_report(USERNAME)

    # Copy the report into the current folder and push it to GitHub.
    copy_and_push_full_report()
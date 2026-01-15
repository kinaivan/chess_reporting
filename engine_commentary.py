import io
import os
from typing import Dict, Optional, List, Tuple

import chess
import chess.pgn
import chess.engine


def _find_engine_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Try to find a usable Stockfish (or compatible UCI) engine binary.

    Order of preference:
    1. Explicit path argument
    2. STOCKFISH_PATH environment variable
    3. A few common default locations
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    common_paths = [
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/opt/homebrew/bin/stockfish",  # common on Apple Silicon
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return None


def _score_to_cp(score: chess.engine.PovScore) -> int:
    """
    Convert a PovScore into a centipawn value, treating mate scores as
    large but clamped values in the appropriate direction.
    """
    if score.is_mate():
        # Treat mate as a big advantage but clamp it so swings stay in a
        # reasonable range for explanation purposes.
        mate_in = score.mate()
        if mate_in is None:
            return 0
        return 800 if mate_in > 0 else -800
    cp = score.score()
    return cp if cp is not None else 0


def generate_engine_commentary_for_game(
    game_data: Dict,
    username: str,
    engine_path: Optional[str] = None,
    depth: int = 12,
) -> str:
    """
    Use a chess engine to analyze a single game and return a short
    commentary string describing the decisive moment.

    The analysis is done from the perspective of `username`. We look for
    the largest evaluation swing in that player's favor (for wins) or
    against them (for losses) and describe that move.
    """
    pgn_text = game_data.get("pgn")
    if not pgn_text:
        return ""

    white = game_data.get("white", {})
    black = game_data.get("black", {})

    white_name = str(white.get("username", "")).lower()
    black_name = str(black.get("username", "")).lower()
    username_lower = username.lower()

    if white_name == username_lower:
        player_color = chess.WHITE
        player_raw_result = str(white.get("result", ""))
    elif black_name == username_lower:
        player_color = chess.BLACK
        player_raw_result = str(black.get("result", ""))
    else:
        # Username not in this game; shouldn't happen, but fail quietly.
        return ""

    # Simplify result into win/loss/draw
    if player_raw_result == "win":
        player_result = "win"
    elif player_raw_result in {
        "agreed",
        "stalemate",
        "repetition",
        "timevsinsufficient",
        "insufficient",
        "50move",
        "draw",
    }:
        player_result = "draw"
    elif player_raw_result in {"checkmated", "resigned", "timeout", "abandoned"}:
        player_result = "loss"
    else:
        player_result = "other"

    engine_exec = _find_engine_path(engine_path)
    if not engine_exec:
        # If no engine is available, don't fail the whole report.
        return ""

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception:
        return ""

    if game is None:
        return ""

    board = game.board()

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_exec)
    except Exception:
        return ""

    eval_history: List[Tuple[int, int, str, bool]] = []
    # (ply_index, eval_cp, san_move, mover_is_player)

    try:
        for ply_index, move in enumerate(game.mainline_moves(), start=1):
            san = board.san(move)
            board.push(move)
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
            except Exception:
                break

            score = info.get("score")
            if score is None:
                continue

            pov_score = score.pov(player_color)
            cp_value = _score_to_cp(pov_score)
            mover_is_player = (board.turn != player_color)  # mover was the side that just played
            eval_history.append((ply_index, cp_value, san, mover_is_player))
    finally:
        try:
            engine.quit()
        except Exception:
            pass

    if len(eval_history) < 2:
        return ""

    # Find the most "decisive" swing.
    best_index = None
    best_magnitude = 0
    prev_cp = eval_history[0][1]

    for i in range(1, len(eval_history)):
        ply_index, curr_cp, san, mover_is_player = eval_history[i]
        delta = curr_cp - prev_cp

        if player_result == "win":
            # Look for big positive swings for the player.
            benefit = delta
            if benefit > 100 and benefit > best_magnitude:
                best_magnitude = benefit
                best_index = i
        elif player_result == "loss":
            # Look for big negative swings on moves played by the player.
            if mover_is_player:
                drop = -delta  # how much worse it got
                if drop > 100 and drop > best_magnitude:
                    best_magnitude = drop
                    best_index = i
        else:
            # For draws/other, look for the biggest absolute swing either way.
            swing = abs(delta)
            if swing > 100 and swing > best_magnitude:
                best_magnitude = swing
                best_index = i

        prev_cp = curr_cp

    if best_index is None:
        # No single "decisive" blunder or winning shot detected.
        if player_result == "win":
            return "You gradually outplayed your opponent without one single decisive blunder."
        if player_result == "loss":
            return "You were gradually outplayed rather than losing to one clear blunder."
        return "The result came from many small shifts rather than a single decisive moment."

    ply_index, cp_after, san, mover_is_player = eval_history[best_index]
    cp_before = eval_history[best_index - 1][1]
    delta_cp = cp_after - cp_before

    swing_pawns = abs(delta_cp) / 100.0

    # Convert from ply index (half-moves) to full-move number so the
    # reported move number matches the PGN move list.
    full_move_no = (ply_index + 1) // 2
    mover_color = "White" if ply_index % 2 == 1 else "Black"

    # Rough phase estimate by full move number.
    if full_move_no <= 10:
        phase = "opening"
    elif full_move_no <= 30:
        phase = "middlegame"
    else:
        phase = "endgame"

    side_str = "you" if mover_is_player else "your opponent"
    move_phrase = f"move {full_move_no} as {mover_color} ({side_str} played {san})"

    big_tactical_swing = swing_pawns >= 2.0

    if player_result == "win":
        if big_tactical_swing:
            if mover_is_player:
                if phase == "opening":
                    return (
                        f"You won largely because of a strong tactical idea in the opening around {move_phrase}, "
                        f"winning material or creating a decisive attack."
                    )
                elif phase == "middlegame":
                    return (
                        f"You won mainly by spotting a tactical shot in the middlegame around {move_phrase}, "
                        f"which gained decisive material."
                    )
                else:
                    return (
                        f"You converted the game in the endgame with precise play around {move_phrase}, "
                        f"turning a balanced position into a winning one."
                    )
            else:
                # Opponent blundered tactically.
                if phase == "opening":
                    return (
                        f"The win came after your opponent made a serious opening mistake around {move_phrase}, "
                        f"allowing you to win material or seize a big initiative."
                    )
                elif phase == "middlegame":
                    return (
                        f"The key reason for the win was a tactical blunder by your opponent in the middlegame "
                        f"around {move_phrase}, after which the position was clearly winning for you."
                    )
                else:
                    return (
                        f"The game swung in your favour in the endgame when your opponent went wrong around "
                        f"{move_phrase}, making it straightforward to convert."
                    )
        # No huge tactical swing: outplaying the opponent.
        if phase == "opening":
            return "You achieved a better position out of the opening and kept control of the game."
        if phase == "middlegame":
            return "You outplayed your opponent in the middlegame, steadily improving your pieces and position."
        return "Your superior endgame technique allowed you to convert the position without a single big blunder."

    if player_result == "loss":
        if big_tactical_swing and mover_is_player:
            if phase == "opening":
                return (
                    f"The loss mainly came from a poor opening decision around {move_phrase}, likely walking into "
                    f"tactics or leaving a piece hanging."
                )
            elif phase == "middlegame":
                return (
                    f"The loss mainly came from a tactical blunder in the middlegame around {move_phrase}, "
                    f"probably hanging material or missing a tactic."
                )
            else:
                return (
                    f"The loss mainly came from inaccurate endgame technique around {move_phrase}, "
                    f"turning a defensible position into a lost one."
                )
        if big_tactical_swing and not mover_is_player:
            if phase == "opening":
                return (
                    f"You got into trouble early when your opponent found a strong idea in the opening around "
                    f"{move_phrase}, after which your position was difficult to hold."
                )
            elif phase == "middlegame":
                return (
                    f"You were outplayed tactically in the middlegame; your opponent's idea around {move_phrase} "
                    f"gave them a decisive advantage."
                )
            else:
                return (
                    f"The endgame slipped away when your opponent found a strong plan around {move_phrase}, "
                    f"making your position very hard to save."
                )

        # No single huge swing: describe as being outplayed in the relevant phase.
        if phase == "opening":
            return "You came out of the opening worse and were never able to fully equalise."
        if phase == "middlegame":
            return "You were gradually outplayed in the middlegame rather than losing to one big blunder."
        return "The endgame was lost through several small inaccuracies rather than a single decisive mistake."

    # Draw or other result.
    if phase == "opening":
        return "The game stayed balanced out of the opening, with no single decisive mistake by either side."
    if phase == "middlegame":
        return "Neither side landed a decisive blow in the middlegame; the game stayed roughly balanced."
    return "The endgame was held by both sides without a clear winning chance for either player."


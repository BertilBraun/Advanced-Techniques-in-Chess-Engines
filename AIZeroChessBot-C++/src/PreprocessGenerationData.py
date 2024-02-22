import os
import chess.pgn
from pathlib import Path

# go through the folder DATABASE_PATH and process all the PGN files
# for each game, extract the moves and the result
# write the moves and the result to a new file in the format:
# SCORE MOVE_1.uci MOVE_2.uci ... MOVE_N.uci


ELITE_GAMES_DOWNLOAD_LINK = "https://player.odycdn.com/v6/streams/b0f01856c521a5f782f8ce4ec6275054e71cf664/3a71ac.mp4?download=true"
EVAL_GAMES_DOWNLOAD_LINK = "https://database.lichess.org/lichess_db_eval.json.zst"


def process_pgn_files(database_path: str, output_file_path: str) -> None:
    total_processed = 0
    
    with open(output_file_path, "w") as output_file:
        for pgn_file in Path(database_path).rglob("*.pgn"):
            print(f"Processing {pgn_file}")
            with open(pgn_file, "r") as file:
                while True:
                    game = chess.pgn.read_game(file)
                    
                    if game is None:
                        break
                    
                    result = game.headers["Result"]
                    output_file.write(f"{result} ")
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        output_file.write(f"{move.uci()} ")
                        
                    output_file.write("\n")
                    total_processed += 1
                    print(f"Processed {total_processed} games", end="\r")
              
    print(f"Processed {total_processed} games")
                    
if __name__ == "__main__":
    database_path = input("Enter the path to the Lichess Elite Database: ")
    output_file = input("Enter the path to the output file: ")

    eval_database_path = input("Enter the path to the Lichess Evaluation Database: ")

    os.makedirs(database_path, exist_ok=True)

    if not os.path.exists(database_path):
        print("Downloading the Lichess Elite Database")
        os.system(f"wget {ELITE_GAMES_DOWNLOAD_LINK} -O data/LichessEliteDatabase.7z")
        os.system("7z x data/LichessEliteDatabase.7z -o data/")
        os.system("rm data/LichessEliteDatabase.7z")  

    if not os.path.exists(eval_database_path):
        print("Downloading the Lichess Evaluation Database")
        os.system(f"wget {EVAL_GAMES_DOWNLOAD_LINK} -O data/LichessEvalDatabase.json.zst")
        os.system("zstd -d data/LichessEvalDatabase.json.zst -o data/LichessEvalDatabase.json")
        os.system(f"mv data/LichessEvalDatabase.json {eval_database_path}")
        os.system("rm data/LichessEvalDatabase.json.zst")
        
    if not os.path.exists(output_file):
        process_pgn_files(database_path, output_file)
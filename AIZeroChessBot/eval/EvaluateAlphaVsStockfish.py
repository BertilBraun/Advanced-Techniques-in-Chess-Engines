from AIZeroChessBot.eval.AlphaZeroBot import AlphaZeroBot

from Framework.Tournament import Tournament
from Framework.ExampleBots.StockfishBot import StockfishBot


def evaluate_alpha_vs_stockfish(network_model_file_path) -> None:
    try:
        alpha_zero = AlphaZeroBot(network_model_file_path)
        stockfish = StockfishBot()

        tournament = Tournament([alpha_zero, stockfish], 10)

        results = tournament.run()

        print('AlphaZero Results:')
        print(results.total_results[alpha_zero])
        print('Stockfish Results:')
        print(results.total_results[stockfish])

        stockfish.cleanup()
    except Exception as e:
        print(f'Error occurred during evaluation of AlphaZero vs Stockfish: {e}')

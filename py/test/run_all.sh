cd ..

python3 -m test.regression_search
#python3 -m test.regression_trainer

python3 -m src.util.remove_repetitions_test

python3 -m src.games.tictactoe.TicTacToe_test
python3 -m src.games.chess.Chess_test

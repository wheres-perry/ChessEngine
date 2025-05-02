import chess
import eval
import load_games


b = load_games.random_board()
e = eval.SimpleEval(b)

print(b)

print(e.basic_evaluate())
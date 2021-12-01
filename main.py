import argparse
from Connect4Graphics import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a game versus AlphaZero.')
    parser.add_argument("-n", "--iterations",type=int, default=100, help="Specify the number of MCTS iterations per move")
    parser.add_argument("-a", "--assist", action="store_true", help="Display the AlphaZero help during the player's turn")
    args = parser.parse_args()
    main(args.iterations, args.assist)
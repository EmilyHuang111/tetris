# main.py

import sys
from game import Game
from custom_model import CUSTOM_AI_MODEL

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [mode]")
        print("Modes: greedy, genetic, mcts, random, student")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "student":
        ai = CUSTOM_AI_MODEL(genotype_file='data/best_genotype.npy')
    else:
        # Initialize other AIs as per existing implementation
        from greedy import Greedy_AI
        from genetic import Genetic_AI
        from mcts import MCTS_AI
        from randomChoice import RandomChoice_NOT_AI

        if mode == "greedy":
            ai = Greedy_AI()
        elif mode == "genetic":
            ai = Genetic_AI()
        elif mode == "mcts":
            ai = MCTS_AI()
        elif mode == "random":
            ai = RandomChoice_NOT_AI()
        else:
            print(f"Unknown mode: {mode}")
            sys.exit(1)

    game = Game(mode, agent=ai)
    if mode == "student":
        game.run_no_visual()
          # Runs with visualization
    else:
        game.run()  # Runs without visualization

if __name__ == "__main__":
    main()

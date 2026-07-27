# neuralswarm.py

import sys
from core.environment import GameEnvironment

if __name__ == "__main__":
    num_agents = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    manual = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
    debug = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True

    env = GameEnvironment(num_agents=num_agents, manual_control=manual, debug=debug, headless=False)
    env.run()

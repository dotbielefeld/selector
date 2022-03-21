import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from selector.scenario import Scenario, parse_args

parser = parse_args()




scenario = Scenario("test_scenario.txt", parser)

print(scenario.parameter)
# -*- coding: utf-8 -*-
"""
This module contains the equilibrium calculation for the JIRA testers report
"""

import itertools
from pprint import pprint
from scipy import stats
import gambit
from gambit.nash import ExternalEnumPureSolver

import game_simulation
import calculate_equilibrium

PLAYER_NUMBER = 5
MAX_RUNS = 100
TIME_FRAMES = 12

# Distribution parameters
ALFA_PARAM = 2.0
BETA_PARAM = 2.0
LAMBDA_PARAM = 3.0

def main():
    """ Initial execution point """
    strategie_values = [(0.0, 0.0),
                        (0.0, 0.5),
                        (0.0, 1.0),
                        (0.5, 0.0), 
                        (0.5, 0.5),                          
                        (0.5, 1.0),                          
                        (1.0, 0.0),                          
                        (1.0, 0.5),
                        (1.0, 1.0)]
    strategy_list = [game_simulation.StochasticInflationStrategy(value, index)
                     for index, value in enumerate(strategie_values)]
                         
    #The profiles are the cartesian product of the possible strategies           
    profiles = list(itertools.product(strategy_list, repeat=PLAYER_NUMBER))
    
    devprod_dist = stats.beta(ALFA_PARAM, BETA_PARAM, loc=0.0, scale=1.0)
    testprod_dist = stats.poisson(LAMBDA_PARAM)
    probability_map = {game_simulation.DEFAULT_KEY: 60.0,
                       game_simulation.SEVERE_KEY: 25.0,
                       game_simulation.NON_SEVERE_KEY: 15.0}
    scores_per_profile = []
    
    for profile in profiles:
        tester_team = []
        for index, strategy in enumerate(profile):
            tester_team.append(game_simulation.Tester("Tester " + str(index), 
                                                      strategy))
        
        scores = game_simulation.simulate(devprod_dist, testprod_dist,
                                          tester_team, probability_map,
                                          TIME_FRAMES, MAX_RUNS)
        pprint(tester_team)     
        print "scores ", scores  
        scores_per_profile.append(scores)

    game = gambit.new_table([len(strategie_values) for _ in range(PLAYER_NUMBER)])
    game.title = "Players: " + str(PLAYER_NUMBER) + " Strategies: " + str(len(strategie_values))

    calculate_equilibrium.define_strategies(game, PLAYER_NUMBER, strategy_list)
    calculate_equilibrium.define_payoffs(game, profiles, scores_per_profile)
    calculate_equilibrium.write_to_file(game=game)
    
#    solver = ExternalEnumPureSolver()
#    result = calculate_equilibrium.compute_nash_equilibrium(game, solver,
#                                                            PLAYER_NUMBER)
    
if __name__ == "__main__":
    main()


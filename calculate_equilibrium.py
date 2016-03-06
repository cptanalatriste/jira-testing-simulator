# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:49:35 2016

@author: Carlos G. Gavidia
"""
import decimal
from pprint import pprint

import gambit
from gambit.nash import ExternalEnumPureSolver
from gambit.nash import ExternalEnumMixedSolver
from gambit.nash import ExternalLogitSolver

import pandas as pd


#TODO(cgavidia): Rename this module

FILE_DIRECTORY = 'C:/Users/cgavi/OneDrive/phd2/jira_data/'
GAME_NAME = 'Estimated_Game'
FILE_NAME = GAME_NAME + '.csv'
OUTPUT_FILE = GAME_NAME + '.nfg'

#Tune this values
PLAYER_NUMBER = 10
STRATEGY_SUBSET = [0, 0.333333333333333, 0.666666666666666]

STRATEGY_INDEX_PREFIX = "Strategy Index for Player "
PAYOFF_VALUE_PREFIX = "Payoff Value for Player "
GAME_TITLE = "The Priority Inflation Game"

def load_dataset():
    """ Loads game information from a CSV file """
    print 'Reading CSV file: ', FILE_NAME
    data_frame = pd.read_csv(FILE_DIRECTORY + FILE_NAME)
    strategy_profiles = data_frame.loc[:, STRATEGY_INDEX_PREFIX + "0" : STRATEGY_INDEX_PREFIX + str(PLAYER_NUMBER - 1)]
    payoffs = data_frame.loc[:, PAYOFF_VALUE_PREFIX + "0" : PAYOFF_VALUE_PREFIX + str(PLAYER_NUMBER - 1)]

    return strategy_profiles, payoffs

def build_strategic_game(strategy_profiles, payoffs):
    """ Generates a Gambit game instance """

    print 'Building game...'
    strategies_list = []
    strategies_per_player = len(STRATEGY_SUBSET)
    for _ in range(PLAYER_NUMBER):
        strategies_list.append(strategies_per_player)

    game = gambit.new_table(strategies_list)
    game.title = GAME_TITLE

    define_strategies(game, PLAYER_NUMBER, STRATEGY_SUBSET)
    define_payoffs_dataframe(game, strategy_profiles, payoffs)
    return game

def define_strategies(game, player_number, strategies):
    """ Assings labels to the Player's strategies """

    print 'Extracting strategies...'
    for player_index in range(player_number):
        strategy_list = game.players[player_index].strategies

        for index, strategy in enumerate(strategy_list):
            strategy.label = str(strategies[index])

def define_payoffs(game, profiles, scores_per_profile):
    """ Configures payoff information in the games"""

    print type(profiles)
    for index, profile in enumerate(profiles):
        profile_payoff = scores_per_profile[index]
        profile_identifier = [strategy.index for strategy in profile]

        for player_index, payoff in enumerate(profile_payoff):
            print 'profile_identifier ', profile_identifier
            print 'player_index ', player_index
            print 'payoff[1]  ', payoff[1]

            #Only to simplify representation
            payoff_as_decimal = int(payoff[1])
            game[profile_identifier][player_index] = payoff_as_decimal

        pprint(profile)
        print profile_payoff

def define_payoffs_dataframe(game, strategy_profiles, payoffs):
    """ Configures the payoff information in the game from a Dataframe"""

    print 'Extracting payoffs ...'
    for index, profile in strategy_profiles.iterrows():
        payoff_vector = payoffs.ix[index]
        index = 0

        for _, payoff_value in payoff_vector.iteritems():
            profile_list = [i.astype(int) for i in profile.tolist()]
            payoff_as_decimal = decimal.Decimal(payoff_value.astype(float))

            game[profile_list][index] = payoff_as_decimal
            index += 1

def list_player_strategies(game, player_index):
    """ List the number of strategies per player """
    strategy_list = game.players[player_index].strategies
    print 'Player ', player_index, ' has ', len(strategy_list), ' strategies.'
    print strategy_list

def list_pure_strategy_profiles(game):
    """ List the pure strategy profiles """
    for profile in game.contingencies:
        payoff_string = ""
        for player_index in range(PLAYER_NUMBER):
            payoff_value = "{0:.3}".format(game[profile][player_index])
            payoff_string += str(payoff_value) + " "

    print "profile ", profile, " payoff_string ", payoff_string

def compute_nash_equilibrium(game, solver, player_number):
    """ Calculates the Nash Equilibrium according to a Solver """
    print 'Computing Nash Equilibrium using: ', type(solver)

    result = solver.solve(game)
    equilibria_found = len(result)
    print 'Equilibria found: ', equilibria_found

    equilibrium_string = ""

    for equilibrium in result:
        equilibrium_profile = equilibrium._profile

        for player_index in range(player_number):
            player = game.players[player_index]
            strategy = equilibrium_profile.__getitem__(player)
            payoff = float(equilibrium_profile.payoff(player))

            equilibrium_string += "Player " + str(player) + ": "
            equilibrium_string += "Strategy " + str(strategy) + " "
            equilibrium_string += " Payoff " + "{0:.3}".format(payoff) + "\t"
        equilibrium_string += "\n"

    print equilibrium_string

    equilibria_file = open(FILE_DIRECTORY + GAME_NAME + str(equilibria_found) + ".txt", "w")
    equilibria_file.write(equilibrium_string)
    equilibria_file.close()

    return result

def write_to_file(output_file=FILE_DIRECTORY + OUTPUT_FILE, game=None):
    """ Writes a configured game in a Gambit-compatible format """

    print 'Writing to game to', output_file
    game_as_file = open(output_file, 'w')
    game_as_file.write(game.write(format='native'))
    game_as_file.close()

def read_game_from_file(game_file):
    """ Reads a Game configuration from a Gambit compatible file """

    print 'Reading game in ', game_file
    return gambit.read_game(game_file)

def main():
    """ Initial execution point """
    strategy_profiles, payoffs = load_dataset()
    game = build_strategic_game(strategy_profiles, payoffs)
    write_to_file(FILE_DIRECTORY + OUTPUT_FILE, game)

    #game = read_game_from_file(file_directory + output_file)

    list_player_strategies(game, 0)
    #list_pure_strategy_profiles(game)

    solver = ExternalEnumPureSolver()
    result = compute_nash_equilibrium(game, solver, PLAYER_NUMBER)

    solver = ExternalEnumMixedSolver()
    result = compute_nash_equilibrium(game, solver, PLAYER_NUMBER)

    solver = ExternalLogitSolver()
    result = compute_nash_equilibrium(game, solver, PLAYER_NUMBER)

if __name__ == "__main__":
    main()



# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:49:35 2016

@author: Carlos G. Gavidia
"""

import gambit
import pandas as pd
import decimal

from gambit.nash import ExternalEnumPureSolver
from gambit.nash import ExternalEnumMixedSolver
from gambit.nash import ExternalLogitSolver

file_directory = 'C:/Users/cgavi/OneDrive/phd2/jira_data/'
game_name = 'Estimated_Game_1454689313717'
file_name =  game_name + '.csv'
output_file = game_name + '.nfg'

#Tune this values
player_number = 10 
strategy_subset = [0, 0.333333333333333, 0.666666666666666]

strategy_index_prefix = "Strategy Index for Player "
payoff_value_prefix = "Payoff Value for Player "
game_title = "The Priority Inflation Game"

def load_dataset():
    print 'Reading CSV file: ', file_name 
    data_frame = pd.read_csv(file_directory + file_name)    
    strategy_profiles = data_frame.loc[:, strategy_index_prefix + "0" : strategy_index_prefix + str(player_number - 1)] 
    payoffs = data_frame.loc[:, payoff_value_prefix + "0" : payoff_value_prefix + str(player_number - 1)] 
    
    return strategy_profiles, payoffs   

def build_strategic_game(strategy_profiles, payoffs):
    print 'Building game...'
    strategies_list = []
    strategies_per_player = len(strategy_subset)
    for strategy_number in range(player_number):
        strategies_list.append(strategies_per_player)
    
    game = gambit.new_table(strategies_list)
    game.title = game_title
    
    define_strategies(game)
    define_payoffs(game, strategy_profiles, payoffs)    
    return game

def define_strategies(game):
    print 'Extracting strategies...'
    for player_index in range(player_number):
        strategy_list = game.players[player_index].strategies
        
        for index, strategy in enumerate(strategy_list):
            strategy.label = "Inflation Ratio " + str(strategy_subset[index])

def define_payoffs(game, strategy_profiles, payoffs):
    print 'Extracting payoffs ...'
    for index, profile in strategy_profiles.iterrows():        
        payoff_vector = payoffs.ix[index]        
        index = 0
        
        for column_name, payoff_value in payoff_vector.iteritems():
            profile_list = [i.astype(int) for i in profile.tolist()] 
            payoff_as_decimal = decimal.Decimal(payoff_value.astype(float))
            
            game[profile_list][index] = payoff_as_decimal            
            index += 1
        
def list_player_strategies(game, player_index):  
    strategy_list = game.players[player_index].strategies     
    print 'Player ', player_index, ' has ', len(strategy_list), ' strategies.'      
    print strategy_list

def list_pure_strategy_profiles(game):
   for profile in game.contingencies:
       payoff_string = ""
       for player_index in range(player_number):
           payoff_value = "{0:.3}".format(game[profile][player_index])
           payoff_string += str(payoff_value) + " "
       
       print "profile ", profile, " payoff_string ", payoff_string

def compute_nash_equilibrium(game, solver):
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
    
    equilibria_file = open(file_directory + game_name + str(equilibria_found) + ".txt", "w")
    equilibria_file.write(equilibrium_string)
    equilibria_file.close()
        
    return result

def write_to_file(output_file, game):
    print 'Writing to game to', output_file
    game_as_file = open(output_file, 'w')
    game_as_file.write(game.write(format='native'))
    game_as_file.close()
    
def read_game_from_file(game_file):
    print 'Reading game in ', output_file
    return gambit.read_game(game_file)

strategy_profiles, payoffs = load_dataset()
game = build_strategic_game(strategy_profiles, payoffs)
write_to_file(file_directory + output_file, game)

#game = read_game_from_file(file_directory + output_file)

list_player_strategies(game, 0)
#list_pure_strategy_profiles(game)

solver = ExternalEnumPureSolver()
result =compute_nash_equilibrium(game, solver)

solver = ExternalEnumMixedSolver()
result = compute_nash_equilibrium(game, solver)

solver = ExternalLogitSolver()
result = compute_nash_equilibrium(game, solver)

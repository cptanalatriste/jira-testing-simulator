""" This module will generate the data files for the simulation statistical analysis
"""

import pandas as pd
import csv
import time

import empirical_study
import game_simulation

FILE_DIRECTORY = 'C:/Users/Carlos G. Gavidia/OneDrive/phd2/jira_data/'


def main():
    """
    Initial execution point.
    """
    boards = [{'id': 2,
               'file': 'UNFILTERED_Tester_Behaviour_Board_2_1457422313707.csv'}]
    simulation_runs = [10, 100, 500, 1000, 5000, 10000]
    runs_per_configuration = 50

    # simulation_runs = [10]
    # runs_per_configuration = 2

    for board in boards:
        board_id = board['id']
        board_file = board['file']

        dataset = pd.read_csv(FILE_DIRECTORY + board_file)
        train_dataset, release_train_dataset, test_dataset, release_test_dataset = empirical_study.split_for_simulation(
            dataset)
        test_releases = len(release_test_dataset.index)

        devprod_dist, testprod_dist, test_team, probability_map = empirical_study.learn_simulation_parameters(
            train_dataset,
            release_train_dataset)

        # TODO(cgavidia): All dataset operations should be on their own file.
        inf_ratio, inflated_issues, sorted_scores = empirical_study.get_inflation_metrics(test_dataset)

        test_best_tester = sorted_scores[0]
        test_runnerup_tester = sorted_scores[1]
        test_worst_tester = sorted_scores[-1]
        test_nexttolast_tester = sorted_scores[-2]

        for simulation_run in simulation_runs:
            file_entries = []
            file_entries.append(("Board", "Simulation Runs", "Index", "Inflation Ratio", "Inflated Issues",
                                 "Top Performer Name", "Top Performer Score", "Runner-up Name", "Runner-up Score",
                                 "Worst Performer Name", "Worst Performer Score", "Next-to-last Name",
                                 "Next-to-last Score", "Inflation Ratio in Test", "Inflated Issues in Test",
                                 "Top Performer Name in Test", "Top Performer Score in Test", "Runner-up Name in Test",
                                 "Runner-up Score in Test",
                                 "Worst Performer Name in Test", "Worst Performer Score in Test",
                                 "Next-to-last Name in Test", "Next-to-last Score in Test", "Execution time"))

            for index in range(runs_per_configuration):
                start_time = time.time()

                total_scores, sim_scores, sim_inflation, sim_ratio = game_simulation.simulate(devprod_dist,
                                                                                              testprod_dist,
                                                                                              test_team,
                                                                                              probability_map,
                                                                                              test_releases,
                                                                                              simulation_run)

                best_tester = sim_scores[0]
                runnerup_tester = sim_scores[1]
                worst_tester = sim_scores[-1]
                nexttolast_tester = sim_scores[-2]
                execution_time = time.time() - start_time

                file_entries.append(
                    (board_id, simulation_run, index, sim_ratio, sim_inflation, best_tester[0], best_tester[1],
                     runnerup_tester[0], runnerup_tester[1], worst_tester[0], worst_tester[1], nexttolast_tester[0],
                     nexttolast_tester[1], inf_ratio, inflated_issues, test_best_tester[0], test_best_tester[1],
                     test_runnerup_tester[0], test_runnerup_tester[1], test_worst_tester[0], test_worst_tester[1],
                     test_nexttolast_tester[0],
                     test_nexttolast_tester[1], execution_time))

            print 'len(file_entries): ', len(file_entries)
            print len(file_entries)
            with open(FILE_DIRECTORY + "Board_" + str(board_id) + " " + str(simulation_run) + "_Simulation_Runs.csv",
                      'wb') as output:
                csv_writer = csv.writer(output, dialect="excel")

                for line in file_entries:
                    csv_writer.writerow(line)


if __name__ == "__main__":
    main()

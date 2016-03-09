""" This module will generate the data files for the simulation statistical analysis
"""

import pandas as pd
import csv
import time

import empirical_study
import game_simulation

FILE_DIRECTORY = 'C:/Users/Carlos G. Gavidia/OneDrive/phd2/jira_data/'


def write_experiment_file(file_name, file_entries):
    """
    Writes the experiment results on a file.
    :param file_name: Name of the file.
    :param file_entries: Entries to write
    :return: None
    """
    with open(file_name, 'wb') as output:
        csv_writer = csv.writer(output, dialect="excel")
        csv_writer.writerow(("Board", "Simulation Runs", "Index", "Inflation Ratio", "Inflated Issues",
                             "Top Performer Name", "Top Performer Score", "Runner-up Name", "Runner-up Score",
                             "Worst Performer Name", "Worst Performer Score", "Next-to-last Name",
                             "Next-to-last Score", "Inflation Ratio in Test", "Inflated Issues in Test",
                             "Top Performer Name in Test", "Top Performer Score in Test", "Runner-up Name in Test",
                             "Runner-up Score in Test",
                             "Worst Performer Name in Test", "Worst Performer Score in Test",
                             "Next-to-last Name in Test", "Next-to-last Score in Test", "Execution time"))

        for line in file_entries:
            csv_writer.writerow(line)


def get_best_worst_information(sorted_scores):
    """
    Given a sorted list of scores, returns the best and worst information as a tuple.
    :param sorted_scores: Sorted list of scores.
    :return: Best-Worst information.
    """
    best_tester = sorted_scores[0]
    runnerup_tester = sorted_scores[1]
    worst_tester = sorted_scores[-1]
    nexttolast_tester = sorted_scores[-2]

    return (best_tester[0], best_tester[1],
            runnerup_tester[0], runnerup_tester[1], worst_tester[0], worst_tester[1],
            nexttolast_tester[0],
            nexttolast_tester[1])


def evaluate_simulation_runs(boards, simulation_runs, runs_per_configuration):
    """
    Executes a series of experiments regarding the number of executions for the Monte Carlo simulation.
    :param boards:  List of bords.
    :param simulation_runs: List of simulation configurations.
    :param runs_per_configuration: How many times each configuration will be run.
    :return:  None
    """
    for board in boards:
        board_id = board['id']
        board_file = board['file']

        dataset = pd.read_csv(FILE_DIRECTORY + board_file)
        train_dataset, release_train_dataset, test_dataset, release_test_dataset = empirical_study.split_for_simulation(
            dataset, board_id)
        test_releases = len(release_test_dataset.index)

        devprod_dist, testprod_dist, test_team, probability_map = empirical_study.learn_simulation_parameters(
            train_dataset,
            release_train_dataset, board_id)

        inf_ratio, inflated_issues, sorted_scores = empirical_study.get_inflation_metrics(test_dataset)
        best_worst_testinfo = get_best_worst_information(sorted_scores)

        for simulation_run in simulation_runs:
            file_entries = []

            for index in range(runs_per_configuration):
                start_time = time.time()

                total_scores, sim_scores, sim_inflation, sim_ratio = game_simulation.simulate(devprod_dist,
                                                                                              testprod_dist,
                                                                                              test_team,
                                                                                              probability_map,
                                                                                              test_releases,
                                                                                              simulation_run)

                best_worst_siminfo = get_best_worst_information(sim_scores)
                execution_time = time.time() - start_time

                entry = (board_id, simulation_run, index, sim_ratio, sim_inflation) + best_worst_siminfo + (
                    inf_ratio, inflated_issues) + best_worst_testinfo + (execution_time,)

                file_entries.append(entry)

            write_experiment_file(
                FILE_DIRECTORY + "Board_" + str(board_id) + " " + str(simulation_run) + "_Simulation_Runs.csv",
                file_entries)


def evaluate_train_split(boards, runs_per_configuration, simulation_run):
    for board in boards:
        board_id = board['id']
        board_file = board['file']

        dataset = pd.read_csv(FILE_DIRECTORY + board_file)

        file_entries = []

        for index in range(runs_per_configuration):
            train_dataset, release_train_dataset, test_dataset, release_test_dataset = empirical_study.split_for_simulation(
                dataset, board_id)
            test_releases = len(release_test_dataset.index)

            devprod_dist, testprod_dist, test_team, probability_map = empirical_study.learn_simulation_parameters(
                train_dataset,
                release_train_dataset, board_id)
            inf_ratio, inflated_issues, sorted_scores = empirical_study.get_inflation_metrics(test_dataset)

            best_worst_testinfo = get_best_worst_information(sorted_scores)
            start_time = time.time()
            total_scores, sim_scores, sim_inflation, sim_ratio = game_simulation.simulate(devprod_dist,
                                                                                          testprod_dist,
                                                                                          test_team,
                                                                                          probability_map,
                                                                                          test_releases,
                                                                                          simulation_run)
            best_worst_siminfo = get_best_worst_information(sim_scores)
            execution_time = time.time() - start_time

            entry = (board_id, simulation_run, index, sim_ratio, sim_inflation) + best_worst_siminfo + (
                inf_ratio, inflated_issues) + best_worst_testinfo + (execution_time,)
            file_entries.append(entry)

        write_experiment_file(
            FILE_DIRECTORY + "Board_" + str(board_id) + "_Test_Train_Split.csv",
            file_entries)


def main():
    """
    Initial execution point.
    """

    boards = [{'id': 2,
               'file': 'UNFILTERED_Tester_Behaviour_Board_2_1457422313707.csv'},
              {'id': 14,
               'file': 'Tester_Behaviour_Board_14_1457459417183.csv'},
              {'id': 83,
               'file': 'Tester_Behaviour_Board_83_1457476536470.csv'}]

    simulation_runs = [10, 50, 100, 250, 500]
    runs_per_configuration = 30

    evaluate_simulation_runs(boards, simulation_runs, runs_per_configuration)
    # evaluate_train_split(boards, 2, 10)


if __name__ == "__main__":
    main()

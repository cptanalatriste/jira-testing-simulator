""" This module is about some charts and metrics about the nature of the data.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/Carlos G. Gavidia/OneDrive/phd2/jira_data/"
ISSUES_FILE = DATA_DIR + "Issues_for_Board_2_1457867608503.csv"
PRIORITY_COLUMN = "Original Priority"
# TODO(cgavidia): The name of this column is misleading and it has a space up front :S
PERIODS_TOFIX_COLUMN = " Releases to Fix (med)"
IGNORED_COLUMN = "Is Ignored"
REJECTED_COLUMN = "Is Rejected"
IMG_DIR = "/plots/"


def issues_by_priority(issues_dataset, priority_list):
    """
    Returns a dataset filtered by priorities.
    :param issues_dataset: Complete dataset.
    :param priority_list: Priorities to include.
    :return: Filtered dataset.
    """
    return issues_dataset[issues_dataset[PRIORITY_COLUMN].isin(priority_list)]


def plot_counts(axis, series, title):
    """
    Plots relative count values for time series.
    :param axis: Axis.
    :param series: Series with data.
    :param title: Char title.
    :return: None.
    """
    axis.set_title(title)
    value_counts = series.value_counts().sort_index()

    value_counts /= float(value_counts.sum())
    value_counts.plot(kind='bar', ax=axis)


def main():
    print "Reading data from CSV ..."
    board_id = 2
    issues_dataset = pd.read_csv(ISSUES_FILE)
    severe_dataset = issues_by_priority(issues_dataset, [1, 2])
    default_dataset = issues_by_priority(issues_dataset, [3])
    nonsevere_dataset = issues_by_priority(issues_dataset, [4, 5])

    print "Plotting periods to fix ..."
    figure, axis = plt.subplots(3, 1, figsize=(30, 15), sharex=False, sharey=True)

    plot_counts(axis[0], severe_dataset[PERIODS_TOFIX_COLUMN],
                "Severe Issues: Periods to fix")
    plot_counts(axis[1], default_dataset[PERIODS_TOFIX_COLUMN],
                "Default Issues: Periods to fix")
    plot_counts(axis[2], nonsevere_dataset[PERIODS_TOFIX_COLUMN],
                "Non-Severe Issues: Periods to fix")

    current_dir = os.path.dirname(__file__)

    plt.savefig(current_dir + IMG_DIR + str(board_id) + '_periods_by_priority.png')

    print "Plotting ignored issues ..."

    figure, axis = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    plot_counts(axis[0], severe_dataset[IGNORED_COLUMN],
                "Severe Issues: Ignored")
    plot_counts(axis[1], default_dataset[IGNORED_COLUMN],
                "Default Issues: Ignored")
    plot_counts(axis[2], nonsevere_dataset[IGNORED_COLUMN],
                "Non-Severe Issues: Ignored")

    plt.savefig(current_dir + IMG_DIR + str(board_id) + '_ignored_by_priority.png')

    print "Plotting rejected issues ..."

    figure, axis = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)
    plot_counts(axis[0], severe_dataset[REJECTED_COLUMN],
                "Severe Issues: Rejected")
    plot_counts(axis[1], default_dataset[REJECTED_COLUMN],
                "Default Issues: Rejected")
    plot_counts(axis[2], nonsevere_dataset[REJECTED_COLUMN],
                "Non-Severe Issues: Rejected")

    plt.savefig(current_dir + IMG_DIR + str(board_id) + '_rejected_by_priority.png')


if __name__ == "__main__":
    main()

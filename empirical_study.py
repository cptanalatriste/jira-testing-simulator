""" This file contains the code required to perform the model validation """

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from collections import defaultdict
from sklearn import cross_validation
import operator

import game_simulation
# TODO(cgavidia): Refactor and rename this module
import data_analysis

FILE_DIRECTORY = 'C:/Users/Carlos G. Gavidia/OneDrive/phd2/jira_data/'
UNFILTERED_FILE_NAME = 'UNFILTERED_Tester_Behaviour_Board_2_1457422313707.csv'
TESTERPROD_COLUMN = "Issues Reported"
TESTERFIX_COLUMN = "Next Release Fixes"
TOT_TESPROD_COLUMN = "Tester Productivity"
NUM_TESTERS_COLUMN = "Number of Testers"
TIMEFRAME_INFLATION_COLUMN = "Possible Inflations"

SEVFIXED_COLUMN = "Severe Release Fixes"
DEFFIXED_COLUMN = "Default Release Fixes"
NONSEVFIXED_COLUMN = "Non Severe Release Fixes"

# Generated columns
AVG_TESTPROD_COLUMN = "Average_Productivity"

# Time frame columns
PERIOD_COLUMN = "Release"
DEVPROD_COLUMN = "Developer Productivity Ratio"

# Tester columns
TESTER_COLUMN = "Tester"
DEFAULT_INFRATIO = "Default Inflation Ratio"
NONSEVERE_INFRATIO = "Non-Severe Inflation Ratio"
TESTER_REPORTS = "Tester Reported Issues"
IMG_DIR = "/plots/"

# Simulation params
MAX_RUNS = 100


def fit_distribution(samples, dist_object, dist_name):
    """ Fits a distribution using maximum likelihood fit and tests it thorugh
    Kolmogorov-Smirnov
    :param dist_name: String identifier of the distribution
    :param dist_object: Potential distribution
    :param samples: Data values """

    dist_params = dist_object.fit(samples)
    print dist_name, ' fitted parameters: ', dist_params
    fitted_dist = dist_object(*dist_params)

    d_stat, p_value = stats.kstest(samples, dist_name, dist_params)
    print dist_name, ' kstest: d ', d_stat, ' p-value ', p_value

    return {"name": dist_name, "dist": fitted_dist, "d": d_stat}


def fit_beta_distribution(samples):
    """ Fits a beta distribution using maximum likelihood fit and tests it thorugh
    Kolmogorov-Smirnov. This includes setting floc and scale parameters and
    adjusting sample information. Keep in mind that this is a standard beta:
    The range is between 0 and 1
    :param samples: Data values"""

    adjusted_samples = samples.copy()
    adjusted_samples[adjusted_samples <= 0] = 0.01
    adjusted_samples[adjusted_samples >= 1] = 0.99

    dist_params = stats.beta.fit(adjusted_samples, floc=0.0, fscale=1.0)

    print 'dist_params', dist_params

    fitted_dist = stats.beta(*dist_params)

    dist_name = "beta"
    d_stat, p_value = stats.kstest(adjusted_samples, dist_name, dist_params)
    print dist_name, ' kstest: d ', d_stat, ' p-value ', p_value

    return {"name": dist_name, "dist": fitted_dist, "d": d_stat}


def continuos_best_fit(samples, board_id=0):
    """ Selects the best-fit distribution for a continuos variable
    :param board_id: Board identifier.
    :param samples: Data values
    """
    normal_dist = fit_distribution(samples, stats.norm, "norm")
    z_stat, p_value = stats.normaltest(samples)
    best_fit = None
    print "Normal test: z ", z_stat, " p_value ", p_value

    # TODO(cgavidia): Normality test is disabled
    p_value = 0

    if p_value > 0.05:
        print "According to normality test, normal is selected"
        best_fit = normal_dist["dist"]

    beta_dist = fit_beta_distribution(samples)

    # uniform_dist = fit_distribution(samples, stats.uniform, "uniform")
    # expon_dist = fit_distribution(samples, stats.expon, "expon")
    #
    # # Results are similar to normal
    # lognorm_dist = fit_distribution(samples, stats.lognorm, "lognorm")
    # gamma_dist = fit_distribution(samples, stats.gamma, "gamma")
    # weibull_min_dist = fit_distribution(samples, stats.weibull_min, "weibull_min")

    dist_list = [  # uniform_dist,
        # normal_dist,
        # expon_dist,
        # lognorm_dist,
        # gamma_dist,
        # weibull_min_dist,
        beta_dist]

    plot_continuos_distributions(samples, dist_list, board_id)

    ks_best_fit = min(dist_list, key=lambda dist: dist["d"])
    print "The best fitting distribution according to kstest ", ks_best_fit["name"]

    if not best_fit and ks_best_fit:
        best_fit = ks_best_fit["dist"]

    return best_fit


def plot_continuos_distributions(samples, dist_list=None, board_id=0):
    """ Plots a data series with a list of fitted distributions
    :param dist_list: List of distributions to include in the plot.
    :param samples: Data values.
    """

    if dist_list is None:
        dist_list = []

    _, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.hist(samples, label="original data", normed=1, bins=10)

    x_values = np.linspace(0, 1)

    for dist in dist_list:
        y_values = dist["dist"].pdf(x_values)
        axis.plot(x_values, y_values, label=dist["name"])

    axis.legend()

    current_dir = os.path.dirname(__file__)
    plt.savefig(current_dir + IMG_DIR + str(board_id) + '_continous_fit.png')


def poisson_best_fit(dataset, board_id=0):
    """ Returns the poisson fit for a sample set
    :param board_id: Board identifier.
    :param dataset: Data values.
    """
    # poisson_model = smf.poisson(AVG_TESTPROD_COLUMN + " ~ 1", data=dataset)
    poisson_model = sm.Poisson(dataset[AVG_TESTPROD_COLUMN], np.ones_like(dataset[AVG_TESTPROD_COLUMN]))

    result = poisson_model.fit()
    lmbda = np.exp(result.params)

    poisson_dist = stats.poisson(lmbda.values)
    lower_poisson_dist = stats.poisson(np.exp(result.conf_int().values)[0, 0])
    higher_poisson_dist = stats.poisson(np.exp(result.conf_int().values)[0, 1])

    print 'result.params ', result.params
    print 'lmbda ', lmbda
    print result.summary()

    testprod_samples = dataset[AVG_TESTPROD_COLUMN]
    print 'testprod_samples.mean ', testprod_samples.mean()

    plot_discrete_distributions(testprod_samples, [{"dist": poisson_dist,
                                                    "color": "red",
                                                    "name": "Poisson Fit"},
                                                   {"dist": lower_poisson_dist,
                                                    "color": "green",
                                                    "name": "Poisson Fit Lower"},
                                                   {"dist": higher_poisson_dist,
                                                    "color": "steelblue",
                                                    "name": "Poisson Fit Higher"}], board_id)

    return poisson_dist


def plot_discrete_distributions(samples, dist_list=None, board_id=0):
    """ Plots a list of discretes distributions
    :param dist_list: List of distributions to include in the plot.
    :param samples: Data values.
    """
    if dist_list is None:
        dist_list = []

    maximum = samples.max()

    hist, bin_edges = np.histogram(samples, range=(0, maximum + 2),
                                   bins=maximum + 2,
                                   normed=True)

    _, axis = plt.subplots(1, 1, figsize=(8, 4))

    axis.bar(bin_edges[:-1], hist, align="center", label="original data")

    for dist in dist_list:
        prob_values = dist["dist"].pmf(bin_edges)
        axis.bar(bin_edges, prob_values, alpha=0.5,
                 width=0.25, color=dist["color"], label=dist["name"])

    axis.legend()

    current_dir = os.path.dirname(__file__)
    plt.savefig(current_dir + IMG_DIR + str(board_id) + '_discrete_fit.png')


def get_release_dataset(dataset):
    """ Produces a release dataset from a dataset of tester reports
    :param dataset: Dataset of tester reports.
    """

    testerprod_series = dataset.groupby(PERIOD_COLUMN)[TESTERPROD_COLUMN].aggregate(np.sum)
    devprod_series = dataset.groupby(PERIOD_COLUMN)[TESTERFIX_COLUMN].aggregate(np.sum)
    testers_series = dataset.groupby(PERIOD_COLUMN)[NUM_TESTERS_COLUMN].aggregate(len)

    release_dataset = pd.concat([testerprod_series, devprod_series, testers_series], axis=1)
    release_dataset[DEVPROD_COLUMN] = release_dataset[TESTERFIX_COLUMN] / release_dataset[TESTERPROD_COLUMN]
    release_dataset[AVG_TESTPROD_COLUMN] = release_dataset[TESTERPROD_COLUMN] / release_dataset[NUM_TESTERS_COLUMN]

    return release_dataset


def get_tester_dataset(dataset):
    """ Produces a tester dataset from a dataset of tester_reports
    :param dataset: Dataset of tester reports.
    """
    tester_dataset = dataset.ix[:, [TESTER_COLUMN, DEFAULT_INFRATIO, TESTER_REPORTS,
                                    NONSEVERE_INFRATIO]]
    tester_dataset = tester_dataset.drop_duplicates()
    tester_dataset = tester_dataset.set_index(TESTER_COLUMN)
    return tester_dataset


def get_inflation_metrics(dataset):
    """ Returns the average inflation and inflated issues for a dataset. It also
    includes the score calculations
    :param dataset: Dataset of tester reports."""

    total_issues = float(dataset[data_analysis.TIMEFRAME_ISSUES_COLUMN].sum())
    inflated_issues = float(dataset[TIMEFRAME_INFLATION_COLUMN].sum())
    inf_ratio = inflated_issues / total_issues
    scores = defaultdict(lambda: 0)
    reports = defaultdict(lambda: 0)

    for index, tester_play in dataset.iterrows():
        tester_name = tester_play[TESTER_COLUMN]
        severe_fixed = tester_play[SEVFIXED_COLUMN]
        default_fixed = tester_play[DEFFIXED_COLUMN]
        nonsevere_fixed = tester_play[NONSEVFIXED_COLUMN]

        reported = tester_play[TESTERPROD_COLUMN]

        score = game_simulation.get_score({game_simulation.SEVERE_KEY: severe_fixed,
                                           game_simulation.DEFAULT_KEY: default_fixed,
                                           game_simulation.NON_SEVERE_KEY: nonsevere_fixed})
        scores[tester_name] += score
        reports[tester_name] += reported

    scores = {tester_name: (float(scores.get(tester_name, 0)) / reports.get(tester_name, 0)
                            if reports.get(tester_name, 0) != 0 else 0.0)
              for tester_name in set(scores)}
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1),
                           reverse=True)

    print 'Dataset: inflation ', inflated_issues
    print 'Dataset ratio ', inf_ratio
    print 'Dataset scores ', sorted_scores

    return inf_ratio, inflated_issues, sorted_scores


def preprocess(dataset, board_id=0):
    """ From the original dataset, this procedure does the following: Removes the 6 more recent time frames and includes
    for analysis only the top 20% of productive tester
    :param board_id: Board identifier.
    :param dataset: Raw dataset
    :return: Polished dataset
    """
    exclusion_list = ["2015-07", "2015-08", "2015-09", "2015-10", "2015-11",
                      "2015-112"]

    tester_dataset = get_tester_dataset(dataset)
    tester_dataset = tester_dataset.sort(TESTER_REPORTS, ascending=False)

    total_testers = len(tester_dataset.index)
    testers_to_include = total_testers / 5

    _, axis = plt.subplots(1, 1, figsize=(16, 8))
    tester_dataset.plot(x=tester_dataset.index, y=TESTER_REPORTS)
    plt.axvline(testers_to_include, color='red', linestyle='--')

    current_dir = os.path.dirname(__file__)
    plt.savefig(current_dir + IMG_DIR + str(board_id) + "_tester_reports.png")

    dataset = dataset[dataset[TESTER_REPORTS] >= tester_dataset.iloc[testers_to_include][TESTER_REPORTS]]
    dataset = dataset[~dataset[PERIOD_COLUMN].isin(exclusion_list)]
    dataset.fillna(0, inplace=True)

    return dataset


def learn_simulation_parameters(train_dataset, release_train_dataset, board_id=0):
    """
    Learns simulation parameter based on the training data provided.
    :param train_dataset: Data set of tester reports.
    :param release_train_dataset: Data set of releases.
    :return: Probability distribution for developer productivity, Probability distribution for Tester Productivity,
    Tester team with strategies, Probability distribution for priorities.
    """
    tester_train_dataset = get_tester_dataset(train_dataset)
    devprod_samples = release_train_dataset[DEVPROD_COLUMN]

    devprod_dist = continuos_best_fit(devprod_samples, board_id)
    testprod_dist = poisson_best_fit(release_train_dataset, board_id)
    test_team = game_simulation.get_tester_team(tester_train_dataset)
    probability_map = data_analysis.get_priority_dictionary(train_dataset)

    return devprod_dist, testprod_dist, test_team, probability_map


def split_for_simulation(dataset, board_id=0):
    """
    Given a dataset is does a random test-train split.
    :param board_id: Board identifier.
    :param dataset: Dataset of tester reports.
    :return: For train and test, a dataset of reports and a dataset of releases.
    """
    dataset = preprocess(dataset, board_id)
    releases = get_release_dataset(dataset)
    release_train_dataset, release_test_dataset = cross_validation.train_test_split(releases,
                                                                                    train_size=0.8)
    train_dataset = dataset[dataset[PERIOD_COLUMN].isin(release_train_dataset.index.values)]
    test_dataset = dataset[dataset[PERIOD_COLUMN].isin(release_test_dataset.index.values)]

    return train_dataset, release_train_dataset, test_dataset, release_test_dataset


def main():
    """ Initial execution point """
    dataset = pd.read_csv(FILE_DIRECTORY + UNFILTERED_FILE_NAME)

    train_dataset, release_train_dataset, test_dataset, release_test_dataset = split_for_simulation(dataset)

    train_releases = len(release_train_dataset.index)
    test_releases = len(release_test_dataset.index)

    print 'train_releases ', train_releases
    print 'test_releases ', test_releases

    devprod_dist, testprod_dist, test_team, probability_map = learn_simulation_parameters(train_dataset,
                                                                                          release_train_dataset)

    print 'probability_map ', probability_map

    # game_simulation.simulate(devprod_dist, testprod_dist, test_team,
    #                          probability_map, train_releases, MAX_RUNS)
    train_avginflation, train_inflation, _ = get_inflation_metrics(train_dataset)

    game_simulation.simulate(devprod_dist, testprod_dist, test_team,
                             probability_map, test_releases, MAX_RUNS)
    avg_inflation, total_inflation, sorted_scores = get_inflation_metrics(test_dataset)


if __name__ == "__main__":
    main()

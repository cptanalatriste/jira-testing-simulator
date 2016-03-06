""" This file contains the code required to perform the model validation """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scipy import stats
from collections import defaultdict
from sklearn import cross_validation
import operator

import game_simulation
#TODO(cgavidia): Refactor and rename this module
import data_analysis

FILE_DIRECTORY = 'C:/Users/cgavi/OneDrive/phd2/jira_data/'
UNFILTERED_FILE_NAME = '10PARTICIPANTS_Tester_Behaviour_Board_2_1457218608670.csv'
TESTER_COLUMN = "Tester"
TESTERPROD_COLUMN = "Issues Reported"
TOT_TESPROD_COLUMN = "Tester Productivity"
NUM_TESTERS_COLUMN = "Number of Testers"
TIMEFRAME_INFLATION_COLUMN = "Possible Inflations"

SEVFIXED_COLUMN = "Severe Release Fixes"
DEFFIXED_COLUMN = "Default Release Fixes"
NONSEVFIXED_COLUMN = "Non Severe Release Fixes"

#Generated columns
YEAR_COLUMN = "Year"
MONTH_COLUMN = "Month"
AVG_TESTPROD_COLUMN = "Average_Productivity"

#Time frame columns
PERIOD_COLUMN = "Release"
DEVPROD_COLUMN = "Developer Productivity Ratio"

#Tester columns
DEFAULT_INFRATIO = "Default Inflation Ratio"
NONSEVERE_INFRATIO = "Non-Severe Inflation Ratio"

#Simulation params
MAX_RUNS = 10000

def fit_distribution(samples, dist_object, dist_name):
    """ Fits a distribution using maximum likelihood fit and tests it thorugh
    Kolmogorov-Smirnov """

    dist_params = dist_object.fit(samples)
    print dist_name, ' fitted parameters: ', dist_params
    fitted_dist = dist_object(*dist_params)

    d_stat, p_value = stats.kstest(samples, dist_name, dist_params)
    print dist_name, ' kstest: d ', d_stat, ' p-value ', p_value

    return {"name": dist_name, "dist": fitted_dist, "d": d_stat}

def fit_beta_distribution(samples):
    """ Fits a beta distribution using maximum likelihood fit and tests it thorugh
    Kolmogorov-Smirnov. This includes setting floc and scale parameters and
    adjusting sample information"""    
    
    adjusted_samples = samples.copy()    
    adjusted_samples[adjusted_samples <= 0] = 0.01
    adjusted_samples[adjusted_samples >= 1] = 0.99    
    
    dist_params = stats.beta.fit(adjusted_samples, floc=0.0,fscale=1.0)
    
    print 'dist_params', dist_params   
    
    fitted_dist = stats.beta(*dist_params)
    
    dist_name = "beta"
    d_stat, p_value = stats.kstest(adjusted_samples, dist_name, dist_params)
    print dist_name, ' kstest: d ', d_stat, ' p-value ', p_value

    return {"name": dist_name, "dist": fitted_dist, "d": d_stat}
    
def continuos_best_fit(samples):
    """ Selects the best-fit distribution for a continuos variable """
    normal_dist = fit_distribution(samples, stats.norm, "norm")
    z_stat, p_value = stats.normaltest(samples)
    best_fit = None
    print "Normal test: z ", z_stat, " p_value ", p_value

    #TODO(cgavidia): Normality test is disabled
    p_value = 0

    if p_value > 0.05:
        print "According to normality test, normal is selected"
        best_fit = normal_dist["dist"]

    uniform_dist = fit_distribution(samples, stats.uniform, "uniform")
    expon_dist = fit_distribution(samples, stats.expon, "expon")

    #Results are similar to normal
    lognorm_dist = fit_distribution(samples, stats.lognorm, "lognorm")
    gamma_dist = fit_distribution(samples, stats.gamma, "gamma")
    beta_dist = fit_beta_distribution(samples)
    weibull_min_dist = fit_distribution(samples, stats.weibull_min, "weibull_min")

    dist_list = [#uniform_dist,
                 #normal_dist,
                 #expon_dist,
                 #lognorm_dist,
                 #gamma_dist,
                 #weibull_min_dist,
                 beta_dist]

    plot_continuos_distributions(samples, dist_list)

    ks_best_fit = min(dist_list, key=lambda dist: dist["d"])
    print "The best fitting distribution according to kstest ", ks_best_fit["name"]

    if not(best_fit) and ks_best_fit:
        best_fit = ks_best_fit["dist"]

    return best_fit

def plot_continuos_distributions(samples, dist_list=None):
    """ Plots a data series with a list of fitted distributions """

    if dist_list is None:
        dist_list = []

    _, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.hist(samples, label="original data", normed=1, bins=10)

    x_values = np.linspace(0, 1)

    for dist in dist_list:
        y_values = dist["dist"].pdf(x_values)
        axis.plot(x_values, y_values, label=dist["name"])

    axis.legend()

def poisson_best_fit(dataset):
    """ Returns the poisson fit for a sample set """
    poisson_model = smf.poisson(AVG_TESTPROD_COLUMN + " ~ 1", data=dataset)

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
                                                   {"dist":lower_poisson_dist,
                                                    "color": "green",
                                                    "name": "Poisson Fit Lower"},
                                                   {"dist": higher_poisson_dist,
                                                    "color": "steelblue",
                                                    "name": "Poisson Fit Higher"}])

    return poisson_dist

def plot_discrete_distributions(samples, dist_list=None):
    """ Plots a list of discretes distributions """
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


def get_release_dataset(dataset):
    """ Produces a release dataset from a dataset of tester reports """
    release_dataset = dataset.ix[:, [PERIOD_COLUMN, DEVPROD_COLUMN,
                                     AVG_TESTPROD_COLUMN]]
    release_dataset = release_dataset.drop_duplicates()
    release_dataset = release_dataset.set_index(PERIOD_COLUMN)

    return release_dataset

def get_tester_dataset(dataset):
    """ Produces a tester dataset from a dataset of tester_reports """
    tester_dataset = dataset.ix[:, [TESTER_COLUMN, DEFAULT_INFRATIO,
                                    NONSEVERE_INFRATIO]]
    tester_dataset = tester_dataset.drop_duplicates()
    tester_dataset = tester_dataset.set_index(TESTER_COLUMN)
    return tester_dataset

def get_inflation_metrics(dataset):
    """ Returns the average inflation and inflated issues for a dataset. It also
    includes the score calculations"""
    total_issues = float(dataset[data_analysis.TIMEFRAME_ISSUES_COLUMN].sum())
    inflated_issues = float(dataset[TIMEFRAME_INFLATION_COLUMN].sum())
    inf_ratio = inflated_issues/total_issues
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
                
    scores = {tester_name: float(scores.get(tester_name, 0))/reports.get(tester_name, 0) 
              for tester_name in set(scores)}
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1),
                           reverse=True)
    
    print 'Dataset: inflation ', inflated_issues
    print 'Dataset ratio ', inf_ratio    
    print 'Dataset scores ', sorted_scores    

    return inf_ratio, inflated_issues, sorted_scores

def main():
    """ Initial execution point """
    dataset = pd.read_csv(FILE_DIRECTORY + UNFILTERED_FILE_NAME)
    exclusion_list = ["2015-07", "2015-08", "2015-09", "2015-10", "2015-11",
                      "2015-112"]
    dataset = dataset[~dataset[PERIOD_COLUMN].isin(exclusion_list)]

    dataset[YEAR_COLUMN] = dataset[PERIOD_COLUMN].apply(lambda period: int(period[:4]))
    dataset[MONTH_COLUMN] = dataset[PERIOD_COLUMN].apply(lambda period: int(period[5:]))
    dataset[AVG_TESTPROD_COLUMN] = dataset[TOT_TESPROD_COLUMN] / dataset[NUM_TESTERS_COLUMN]
    dataset[AVG_TESTPROD_COLUMN] = dataset[AVG_TESTPROD_COLUMN].astype(int)
    dataset.fillna(0, inplace=True)
    
    releases = get_release_dataset(dataset)    
    release_train_dataset, release_test_dataset = cross_validation.train_test_split(releases,
                                                                                    train_size=0.5)
    train_dataset = dataset[dataset[PERIOD_COLUMN].isin(release_train_dataset.index.values)]     
    test_dataset = dataset[dataset[PERIOD_COLUMN].isin(release_test_dataset.index.values)]     
    train_releases = len(release_train_dataset.index)
    test_releases = len(release_test_dataset.index)

    print 'train_releases ', train_releases
    print 'test_releases ', test_releases  

    tester_train_dataset = get_tester_dataset(train_dataset)

    devprod_samples = release_train_dataset[DEVPROD_COLUMN]
    print devprod_samples.unique()
   
    devprod_dist = continuos_best_fit(devprod_samples)
    testprod_dist = poisson_best_fit(release_train_dataset)
    test_team = game_simulation.get_tester_team(tester_train_dataset)
    probability_map = data_analysis.get_priority_dictionary(train_dataset)

#    game_simulation.simulate(devprod_dist, testprod_dist, test_team,
#                             probability_map, train_releases, MAX_RUNS)
    train_avginflation, train_inflation, train_scores = get_inflation_metrics(train_dataset)

    game_simulation.simulate(devprod_dist, testprod_dist, test_team,
                             probability_map, test_releases, MAX_RUNS)
    avg_inflation, total_inflation, sorted_scores = get_inflation_metrics(test_dataset)

if __name__ == "__main__":
    main()

""" The data_analysis modules extract some information regarding the datasets
containing the test report information."""

import pandas as pd
from pandas import DataFrame

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from scipy import stats

# File location constants
FILE_DIRECTORY = 'C:/Users/cgavi/OneDrive/phd2/jira_data/'
FILE_NAME = 'Tester_Behaviour_Board_2_1455916587391.csv'
ISSUES_FILE_NAME = 'Issues_for_Board_2_1455828073134.csv'


TESTERS = ['rayeesn', 'sangeethah', 'chandanp', 'jessicawang', 'sailaja',
           'parth', 'minchen07', 'alena1108', 'nitinme']
DATA_COLUMNS = ['Expected Inflated Fixes', 'Expected Severe Fixes',
                'Expected Non Severe Fixes']

TARGET_COLUMN = 'Next Release Fixes'
STRATEGY_COLUMN = 'Inflation Ratio'
DEV_PROD_COLUMN = 'Developer Productivity Ratio'
TESTER_PROD_COLUMN = 'Tester Productivity'
SEVERE_FOUND_COLUMN = 'Severe Issues'
NONSEVERE_FOUND_COLUMN = 'Non-Severe Issues Found'
SUCCESS_COLUMN = 'Success Ratio'
PREVIOUS_SUCCESS_COLUMN = 'Previous Success Ratio'
SEVERE_REPORTED_COLUMN = 'Severe Ratio Reported'
TIMEFRAME_ISSUES_COLUMN = 'Issues Reported'
TIMEFRAME_SEVERE_COLUMN = 'Severe Issues'
TIMEFRAME_NONSEVERE_COLUMN = 'Non-Severe Issues Found'
TIMEFRAME_DEFAULT_COLUMN = 'Default Issues Found'
INF_RATIO_COLUMN = 'Time Frame Inflation Ratio'

#Issue CSV columns
PRIORITY_COLUMN = 'Original Priority'
RELEASES_TOFIX_COLUMN = ' Releases to Fix (med)'


def load_issues_dataset():
    """ Loads the CSV containing detailed issue information """
    issues_data_frame = pd.read_csv(FILE_DIRECTORY + ISSUES_FILE_NAME)
    return issues_data_frame

def load_game_dataset():
    """ Loads the data set related to tester behaviour over games"""

    data_frame = pd.read_csv(FILE_DIRECTORY + FILE_NAME)

    print 'MAX: Next Release Fixes', np.max(data_frame[TARGET_COLUMN])
    print 'MIN: Next Release Fixes', np.min(data_frame[TARGET_COLUMN])
    print 'MEAN: Next Release Fixes', np.mean(data_frame[TARGET_COLUMN])

    print 'MAX: ' + STRATEGY_COLUMN, np.max(data_frame[STRATEGY_COLUMN])
    print 'MIN: ' + STRATEGY_COLUMN, np.min(data_frame[STRATEGY_COLUMN])
    print 'MEAN: ' + STRATEGY_COLUMN, np.mean(data_frame[STRATEGY_COLUMN])

    return data_frame

def split_dataset(data_frame, scale=True):
    """ Splits the dataset between train and test subsets """

    data = data_frame.loc[:, DATA_COLUMNS]
    target = data_frame.loc[:, TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.25,
                                                        random_state=33)

    if scale:
        print 'Scaling data...'
        scaler_x = StandardScaler().fit(x_train)
        scaler_y = StandardScaler().fit(y_train)

        x_train = scaler_x.transform(x_train)
        y_train = scaler_y.transform(y_train)
        x_test = scaler_x.transform(x_test)
        y_test = scaler_y.transform(y_test)

    return x_train, y_train, x_test, y_test

def train_and_evaluate(regressor, x_train, y_train):
    """ Trains a regressor and evaluates the goodness of the fit """

    regressor.fit(x_train, y_train)
    print 'Coefficient of determination on training set: ', regressor.score(x_train,
                                                                            y_train)
    cross_validation = KFold(x_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(regressor, x_train, y_train, cv=cross_validation)

    print 'scores', scores
    print 'Average coefficient of determination using 5-fold cross-validation', np.mean(scores)

def create_linear_regressor(x_train, y_train):
    """ Creates a linear regresor based on train data """

    regressor = linear_model.SGDRegressor()
    train_and_evaluate(regressor, x_train, y_train)
    return regressor

def create_svm_regressor(x_train, y_train):
    """ Creates a linear regressor based on SVR """

    regressor = svm.SVR(kernel='rbf')
    train_and_evaluate(regressor, x_train, y_train)
    return regressor

def measure_performance(x_test, y_test, regressor):
    """ Evaluates the regressor performance on the Test dataset """

    y_pred = regressor.predict(x_test)

    print 'Coefficient of determination: {0:.3f}'.format(
        metrics.r2_score(y_test, y_pred)), '\n'

def plot_external_event(data_frame, event_column, axis=None, prefix='', bins=25):
    """ Plots a histogram of an specific column on the dataset """

    values = data_frame[event_column]
    values.hist(bins=bins, alpha=0.3, normed=True, ax=axis)
    values.plot(kind='kde', style='k--', ax=axis, title=prefix + event_column)

    #values.hist(bins=50, alpha=0.3,ax=ax)

def plot_strategy(data_frame, x_column=None, y_column=None, axis=None, title=None):
    """ Plots the relationship between two columns on the dataset """
    data_frame.plot(kind='line', x=x_column, y=y_column, ax=axis, title=title)

def plot_tester_strategy(tester_list, data_frame, metrics):
    """ Plots a list of metrics for a list of testers """

    for tester_name in tester_list:
        tester_data_frame = load_tester_reports(data_frame, tester_name)
        width_height = (16, 8)
        _, axes = plt.subplots(nrows=1, ncols=1, figsize=width_height)

        for _, metric in enumerate(metrics):
            plot_strategy(axis=axes, data_frame=tester_data_frame, x_column='Release',
                          y_column=metric, title=metric + " - "+ tester_name)

def calculate_correlations(testers, data_frame, first_column, second_column):
    """ For a list of testers, calculates the correlation between
    two attributes """

    print 'Exploring correlation between ', first_column, '- ', second_column

    for tester_name in testers:
        tester_data_frame = load_tester_reports(data_frame, tester_name)
        (correlation, tail) = stats.pearsonr(tester_data_frame[first_column],
                                             tester_data_frame[second_column])

        message = tester_name + ': ' + str(correlation) + ' ' + str(tail)
        print message
        tester_data_frame.plot(x=first_column, y=second_column, style='o',
                               title=message)

def sample_tester_productivity(data_frame):
     """ Returns all the data points regarding issues reported per time frame
     per tester """
     return data_frame[TIMEFRAME_ISSUES_COLUMN]

def sample_dev_productivity(release_data_frame):
    """ Returns all the data points regarding developer productivity ratios """
    return release_data_frame[DEV_PROD_COLUMN]

def sample_severe_found(data_frame):
    """ Returns all the datapoints related to the severe issues found by a
    tester """
    return data_frame[SEVERE_FOUND_COLUMN]

def sample_nonsevere_found(data_frame):
    """ Returns all the datapoints related to the non-severe issues found by
    a tester """
    return data_frame[NONSEVERE_FOUND_COLUMN]

def load_release_dataset(data_frame):
    """ Produces a dataframe consolidating release information """
    release_values = data_frame['Release'].unique()
    developer_productivity_values = []
    dev_productivity_ratio_values = []
    avg_inflation_ratio = []
    var_inflation_ratio = []
    med_inflation_ratio = []
    severity_ratio = []
    number_of_testers = []
    tester_productivity = []
    timeframe_inf_ratio = []

    for release in release_values:
        release_data = data_frame[data_frame['Release'].isin([release])]
        developer_productivity_values.append(release_data['Developer Productivity'].iloc[0])
        dev_productivity_ratio_values.append(release_data[DEV_PROD_COLUMN].iloc[0])
        avg_inflation_ratio.append(release_data['Inflation Ratio (mean)'].iloc[0])
        med_inflation_ratio.append(release_data['Inflation Ratio (med)'].iloc[0])
        var_inflation_ratio.append(release_data['Inflation Ration (var)'].iloc[0])
        severity_ratio.append(release_data['Release Severity Ratio'].iloc[0])
        number_of_testers.append(release_data['Number of Testers'].iloc[0])
        tester_productivity.append(release_data[TESTER_PROD_COLUMN].iloc[0])
        timeframe_inf_ratio.append(release_data[INF_RATIO_COLUMN].iloc[0])

    return DataFrame({'Order': range(len(release_values)),
                      'Release': release_values,
                      'Developer Productivity': developer_productivity_values,
                      DEV_PROD_COLUMN: dev_productivity_ratio_values,
                      'Inflation Ratio (mean)': avg_inflation_ratio,
                      'Inflation Ratio (med)': med_inflation_ratio,
                      'Inflation Ration (var)': var_inflation_ratio,
                      'Release Severity Ratio': severity_ratio,
                      'Number of Testers': number_of_testers,
                      TESTER_PROD_COLUMN: tester_productivity,
                      INF_RATIO_COLUMN: timeframe_inf_ratio})


def load_tester_reports(data_frame, tester_name):
    """ Generates a data frame related to a specific tester, including the
    Previous Success Ratio column """

    tester_data_frame = data_frame[data_frame['Tester'].isin([tester_name])]
    tester_data_frame[PREVIOUS_SUCCESS_COLUMN] = tester_data_frame[SUCCESS_COLUMN].shift(1)
    tester_data_frame[PREVIOUS_SUCCESS_COLUMN].fillna(0, inplace=True)

    return tester_data_frame

def plot_severity_fixes(issues_data_frame, priority):
    """ For a severity level, plots the number of releases required to be
    fixed """

    nonseverity_data_frame = issues_data_frame[issues_data_frame[PRIORITY_COLUMN].isin([priority])]
    nonseverity_data_frame[RELEASES_TOFIX_COLUMN].fillna(-4, inplace=True)

    width_height = (16, 8)
    _, axis = plt.subplots(nrows=1, ncols=1, figsize=width_height)

    plot_external_event(nonseverity_data_frame, RELEASES_TOFIX_COLUMN, axis,
                        'Priority-' + str(priority) + ' ')

def get_priority_dictionary(data_frame):
    """ Calculates probabilities for each priority """
    total_issues = float(data_frame[TIMEFRAME_ISSUES_COLUMN].sum())
    total_severe = data_frame[TIMEFRAME_SEVERE_COLUMN].sum()
    total_nonsevere = data_frame[TIMEFRAME_NONSEVERE_COLUMN].sum()
    total_default = data_frame[TIMEFRAME_DEFAULT_COLUMN].sum()

    return {'severe': total_severe/total_issues,
            'non_severe': total_nonsevere/total_issues,
            'default': total_default/total_issues}
def main():
    """ Start point for the module"""
    data_frame = load_game_dataset()
    issues_data_frame = load_issues_dataset()
    release_data_frame = load_release_dataset(data_frame)

    #Plotting external event data
    width_height = (20, 60)
    _, axes = plt.subplots(12, 1, figsize=width_height)
    plot_external_event(data_frame, STRATEGY_COLUMN, axes[0])
    plot_external_event(data_frame, SEVERE_FOUND_COLUMN, axes[1])
    plot_external_event(data_frame, NONSEVERE_FOUND_COLUMN, axes[2])
    plot_external_event(release_data_frame, DEV_PROD_COLUMN, axes[3], bins=5)
    plot_external_event(release_data_frame, 'Release Severity Ratio', axes[4])
    plot_external_event(release_data_frame, 'Number of Testers', axes[5])
    plot_external_event(release_data_frame, TESTER_PROD_COLUMN, axes[6])

    plot_strategy(release_data_frame, 'Release', 'Inflation Ratio (mean)', axes[7])
    plot_strategy(release_data_frame, 'Release', 'Inflation Ratio (med)', axes[8])
    plot_strategy(release_data_frame, 'Release', 'Inflation Ration (var)', axes[9])
    plot_external_event(data_frame, TIMEFRAME_ISSUES_COLUMN, axes[10])
    plot_strategy(release_data_frame, 'Release', INF_RATIO_COLUMN, axes[11])

    plot_tester_strategy(TESTERS, data_frame, [STRATEGY_COLUMN, SUCCESS_COLUMN,
                                               SEVERE_REPORTED_COLUMN])
    calculate_correlations(TESTERS, data_frame, PREVIOUS_SUCCESS_COLUMN,
                           STRATEGY_COLUMN)


    #Creating regression
    x_train, y_train, x_test, y_test = split_dataset(data_frame, False)

    regressor = create_linear_regressor(x_train, y_train)
    print 'regressor.coef_', regressor.coef_
    print 'regressor.intercept_ ', regressor.intercept_

    measure_performance(x_test, y_test, regressor)

    for priority in range(1, 6):
        plot_severity_fixes(issues_data_frame, priority)

if __name__ == '__main__':
    main()





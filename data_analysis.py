import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import *
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
from scipy import stats
from pandas import DataFrame

file_directory = 'C:/Users/cgavi/OneDrive/phd2/jira_data/'
file_name = 'Tester_Behaviour_Board_2_1455831954061.csv'
issues_file_name = 'Issues_for_Board_2_1455828073134.csv'


testers = ['rayeesn', 'sangeethah', 'chandanp', 'jessicawang', 'sailaja',
              'parth','minchen07', 'alena1108', 'nitinme', 'minchen07']
data_columns = ['Expected Inflated Fixes', 'Expected Severe Fixes',
                'Expected Non Severe Fixes']

#TODO(cgavidia): Change caps for String constants
target_column = 'Next Release Fixes'
strategy_column = 'Inflation Ratio'
dev_prod_column = 'Developer Productivity Ratio'
tester_prod_column = 'Tester Productivity'
severe_found_column = 'Severe Issues'
nonsevere_found_column = 'Non-Severe Issues Found'
success_column = 'Success Ratio'
previous_success_column = 'Previous Success Ratio'
severe_reported_column = 'Severe Ratio Reported'
time_frame_issues_column = 'Issues Reported'
time_frame_severe_column = 'Severe Issues'
time_frame_nosevere_column = 'Non-Severe Issues Found'
time_frame_default_column = 'Default Issues Found'

#Issue CSV columns
priotity_column = 'Original Priority'
releases_to_fix_column = ' Releases to Fix (med)'


def load_issues_dataset():
    """ Loads the CSV containing detailed issue information """
    issues_data_frame = pd.read_csv(file_directory + issues_file_name)
    return issues_data_frame

def load_game_dataset():
    """ Loads the data set related to tester behaviour over games"""

    data_frame = pd.read_csv(file_directory + file_name)

    print 'MAX: Next Release Fixes', np.max(data_frame[target_column])
    print 'MIN: Next Release Fixes', np.min(data_frame[target_column])
    print 'MEAN: Next Release Fixes', np.mean(data_frame[target_column])

    print 'MAX: ' + strategy_column, np.max(data_frame[strategy_column])
    print 'MIN: ' + strategy_column, np.min(data_frame[strategy_column])
    print 'MEAN: ' + strategy_column, np.mean(data_frame[strategy_column])

    return data_frame

def split_dataset(data_frame, scale=True):
    data = data_frame.loc[:, data_columns]
    target = data_frame.loc[:, target_column]

    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size = 0.25,
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
    regressor.fit(x_train, y_train)
    print 'Coefficient of determination on training set: ', regressor.score(x_train,
                                                                            y_train)
    cv = KFold(x_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(regressor, x_train, y_train, cv=cv)

    print 'scores', scores
    print 'Average coefficient of determination using 5-fold cross-validation', np.mean(scores)

def create_linear_regressor(x_train, y_train):
    regressor = linear_model.SGDRegressor()
    train_and_evaluate(regressor, x_train, y_train)
    return regressor

def create_svm_regressor(x_train, y_train):
    regressor = svm.SVR(kernel='rbf')
    train_and_evaluate(regressor, x_train, y_train)
    return regressor

def measure_performance(x_test, y_test, regressor):
    y_pred = regressor.predict(x_test)

    print 'Coefficient of determination: {0:.3f}'.format(
        metrics.r2_score(y_test, y_pred)), '\n'

def plot_external_event(data_frame, event_column, ax=None, prefix='', bins=25):
    values = data_frame[event_column]
    values.hist(bins=bins, alpha=0.3, normed=True, ax=ax)
    values.plot(kind='kde', style='k--', ax=ax, title=prefix + event_column)

    #values.hist(bins=50, alpha=0.3,ax=ax)

def plot_strategy(data_frame, x_column=None, y_column=None, ax=None, title=None):
    data_frame.plot(kind='line', x=x_column, y=y_column, ax=ax, title=title)

def plot_tester_strategy(tester_list, data_frame, metrics):
    for tester_name in tester_list:
        tester_data_frame = load_tester_reports(data_frame, tester_name)
        width_height = (16, 8)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=width_height)

        for index, metric in enumerate(metrics):
            plot_strategy(ax=axes, data_frame=tester_data_frame, x_column='Release',
              y_column=metric, title= metric + " - "+ tester_name)

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
     return data_frame[time_frame_issues_column]

def sample_dev_productivity(release_data_frame):
    """ Returns all the data points regarding developer productivity ratios """
    return release_data_frame[dev_prod_column]

def sample_severe_found(data_frame):
    """ Returns all the datapoints related to the severe issues found by a
    tester """
    return data_frame[severe_found_column]

def sample_nonsevere_found(data_frame):
    """ Returns all the datapoints related to the non-severe issues found by
    a tester """
    return data_frame[nonsevere_found_column]

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

    for release in release_values:
        release_data = data_frame[data_frame['Release'].isin([release])]
        developer_productivity_values.append(release_data['Developer Productivity'].iloc[0])
        dev_productivity_ratio_values.append(release_data[dev_prod_column].iloc[0])
        avg_inflation_ratio.append(release_data['Inflation Ratio (mean)'].iloc[0])
        med_inflation_ratio.append(release_data['Inflation Ratio (med)'].iloc[0])
        var_inflation_ratio.append(release_data['Inflation Ration (var)'].iloc[0])
        severity_ratio.append(release_data['Release Severity Ratio'].iloc[0])
        number_of_testers.append(release_data['Number of Testers'].iloc[0])
        tester_productivity.append(release_data[tester_prod_column].iloc[0])

    return DataFrame({'Order': range(len(release_values)),
                      'Release': release_values,
                      'Developer Productivity': developer_productivity_values,
                      dev_prod_column: dev_productivity_ratio_values,
                      'Inflation Ratio (mean)': avg_inflation_ratio,
                      'Inflation Ratio (med)': med_inflation_ratio,
                      'Inflation Ration (var)': var_inflation_ratio,
                      'Release Severity Ratio': severity_ratio,
                      'Number of Testers': number_of_testers,
                      tester_prod_column: tester_productivity})


def load_tester_reports(data_frame, tester_name):
    """ Generates a data frame related to a specific tester, including the
    Previous Success Ratio column """

    tester_data_frame = data_frame[data_frame['Tester'].isin([tester_name])]
    tester_data_frame[previous_success_column] =  tester_data_frame[success_column].shift(1)
    tester_data_frame[previous_success_column].fillna(0, inplace=True)

    return tester_data_frame

def plot_severity_fixes(issues_data_frame, priority):
    nonseverity_data_frame = issues_data_frame[issues_data_frame[priotity_column].isin([priority])]
    nonseverity_data_frame[releases_to_fix_column].fillna(-4, inplace=True)

    width_height = (16, 8)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=width_height)

    plot_external_event(nonseverity_data_frame, releases_to_fix_column, ax,
                        'Priority-' + str(priority) + ' ')

def get_priority_dictionary(data_frame):
    """ Calculates probabilities for each priority """
    total_issues = float(data_frame[time_frame_issues_column].sum())
    total_severe = data_frame[time_frame_severe_column].sum()
    total_nonsevere = data_frame[time_frame_nosevere_column].sum()
    total_default = data_frame[time_frame_default_column].sum()

    return {'severe': total_severe/total_issues,
            'non_severe': total_nonsevere/total_issues,
            'default': total_default/total_issues}

if __name__ == '__main__':
    data_frame = load_game_dataset()
    issues_data_frame = load_issues_dataset()
    release_data_frame = load_release_dataset(data_frame)

    #Plotting external event data
    width_height = (20, 60)
    fig, axes = plt.subplots(11, 1, figsize=width_height)
    plot_external_event(data_frame, strategy_column, axes[0])
    plot_external_event(data_frame, severe_found_column, axes[1])
    plot_external_event(data_frame, nonsevere_found_column, axes[2])
    plot_external_event(release_data_frame, dev_prod_column, axes[3], bins=5)
    plot_external_event(release_data_frame, 'Release Severity Ratio', axes[4])
    plot_external_event(release_data_frame, 'Number of Testers', axes[5])
    plot_external_event(release_data_frame, tester_prod_column, axes[6])

    plot_strategy(release_data_frame, 'Release', 'Inflation Ratio (mean)', axes[7])
    plot_strategy(release_data_frame, 'Release', 'Inflation Ratio (med)', axes[8])
    plot_strategy(release_data_frame, 'Release', 'Inflation Ration (var)', axes[9])
    plot_external_event(data_frame, time_frame_issues_column, axes[10])

    plot_tester_strategy(testers, data_frame, [strategy_column, success_column,
                                               severe_reported_column])
    calculate_correlations(testers, data_frame, previous_success_column,
                           strategy_column)


    #Creating regression
    x_train, y_train, x_test, y_test = split_dataset(data_frame, False)

    regressor = create_linear_regressor(x_train, y_train)
    print 'regressor.coef_', regressor.coef_
    print 'regressor.intercept_ ', regressor.intercept_

    measure_performance(x_test, y_test, regressor)

    for priority in range(1, 6):
        plot_severity_fixes(issues_data_frame, priority)





""" Support for the multi-agent simulation """
import random

from collections import defaultdict
import operator

from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import data_analysis

TIME_FRAMES = 1000

SEVERE_KEY = 'severe'
DEFAULT_KEY = 'default'
NON_SEVERE_KEY = 'non_severe'

REPORTED_SUFFIX = '_reported'
FIXED_SUFFIX = '_fixed'
INFLATED_SUFFIX = '_inflated'

SEVERE_VALUE = 7
DEFAULT_VALUE = 3
NONSEVERE_VALUE = 1

class DeveloperTeam(object):
    """ A Developer Team that fixes defects based on reporter priorities """

    def __init__(self, cont_dist):
        """ According to latest data analysis, the productivity takes a
        Continuous distribution """
        self.cont_dist = cont_dist

    def fix(self, tester_reports):
        """ Giving a list of reports, it returns how many of them got fixed """

        print 'tester_reports ', tester_reports
        productivity = self.get_developer_productivity(tester_reports)
        fix_reports = [{SEVERE_KEY: 0,
                        NON_SEVERE_KEY: 0,
                        DEFAULT_KEY: 0}
                       for _ in range(len(tester_reports))]

        severe_reports = []
        default_reports = []
        non_severe_reports = []

        #TODO(cgavidia): We should find a more elegant way to deal with priorities
        for tester_index, report in enumerate(tester_reports):
            severe_reports.extend([tester_index for _ in range(report[SEVERE_KEY])])
            default_reports.extend([tester_index for _ in range(report[DEFAULT_KEY])])
            non_severe_reports.extend([tester_index for _ in range(report[NON_SEVERE_KEY])])

        severe_fixed = self.process_priority_batch(productivity, severe_reports,
                                                   fix_reports, SEVERE_KEY)
        default_fixed = 0

        if productivity - severe_fixed > 0:
            default_fixed = self.process_priority_batch(productivity - severe_fixed,
                                                        default_reports, fix_reports,
                                                        DEFAULT_KEY)

        if productivity - severe_fixed - default_fixed > 0:
            self.process_priority_batch(productivity - severe_fixed - default_fixed,
                                        non_severe_reports, fix_reports,
                                        NON_SEVERE_KEY)

        # TODO(cgavidia): Big assumption! The severe reports that didn't get to fixed
        # are ignored so far. We not considerer the possibility to be fixed later.
        print 'fix_reports ', fix_reports
        return fix_reports

    def get_developer_productivity(self, tester_reports):
        """ Calculates the number of fixes that will be delivered on a time frame"""

        consolidated_reports = [report[SEVERE_KEY] + report[NON_SEVERE_KEY] + report[DEFAULT_KEY]
                                for report in tester_reports]

        defects_reported = sum(consolidated_reports)
        from_distribution = self.cont_dist.rvs()
        print 'defects_reported ', defects_reported

        productivity_ratio = from_distribution
        productivity = np.random.binomial(defects_reported, productivity_ratio)

        print 'productivity_ratio ', productivity_ratio
        print 'productivity ', productivity

        return productivity

    def process_priority_batch(self, productivity, issue_list, fix_reports,
                               priority_index):
        """ Process an batch of issues. It returns the number of fixes delivered """

        random.shuffle(issue_list)

        fixes_delivered = 0
        for tester_index in issue_list:
            if fixes_delivered < productivity:
                fixes_for_tester = fix_reports[tester_index]
                fixes_for_tester[priority_index] += 1
                fixes_delivered += 1
            else:
                break

        return fixes_delivered

class StochasticInflationStrategy(object):
    """ Defines a particular strategy for testing on a release """

    def __init__(self, inflation_prob=(0, 0), index=0):
        self.for_default = inflation_prob[0]
        self.for_nonsevere = inflation_prob[1]
        self.index = index

    def report(self, issues_found):
        """ Makes a report, with the inflation results based on probabilities.
        All inflations are promotions to severe """

        severe_reported = issues_found[SEVERE_KEY]
        default_reported = issues_found[DEFAULT_KEY]
        nonsevere_reported = issues_found[NON_SEVERE_KEY]

        default_inflation = self.inflate(issues_found[DEFAULT_KEY],
                                         self.for_default)
        nonsevere_inflation = self.inflate(issues_found[NON_SEVERE_KEY],
                                           self.for_nonsevere)

        severe_reported += default_inflation + nonsevere_inflation
        default_reported -= default_inflation
        nonsevere_reported -= nonsevere_reported

        inflation = {DEFAULT_KEY: default_inflation,
                     NON_SEVERE_KEY: nonsevere_inflation}
        report = {SEVERE_KEY: severe_reported,
                  DEFAULT_KEY: default_reported,
                  NON_SEVERE_KEY: nonsevere_reported}

        return inflation, report

    def inflate(self, issues, inflation_prob):
        """ Return the number of issues to inflate based on the probability """

        inflated_issues = 0

        for _ in range(issues):
            if random.uniform(0, 1) < inflation_prob:
                inflated_issues += 1

        return inflated_issues

    def __str__(self):
        return "D " + str(self.for_default) + \
                ", NS " + str(self.for_nonsevere)

    def __repr__(self):
        return str(self)

class Tester(object):
    """ A Tester, that reports defect for a Software Version """

    def __init__(self, name, testing_strategy):
        self.name = name
        self.testing_strategy = testing_strategy
        self.reset()

    def reset(self):
        """ Clears the testing history """

        self.release_reports = []
        self.inflation_reports = []
        self.fix_reports = []
        self.scores = []

    def report(self, issue_list):
        """ Reports a number of defects on the System """

        inflation, report = self.testing_strategy.report(issue_list)
        self.record(test_report=report, inflation_report=inflation)
        return report

    def consolidate_release_reports(self):
        """ Calculates the total number of issues reported by release """
        return [report[SEVERE_KEY] + report[NON_SEVERE_KEY] + report[DEFAULT_KEY]
                for report in self.release_reports]

    def record(self, test_report=None, inflation_report=None, fix_report=None):
        """ Stores the report made """

        if test_report:
            self.release_reports.append(test_report)

        if inflation_report:
            self.inflation_reports.append(inflation_report)

        if fix_report:
            self.fix_reports.append(fix_report)
            self.scores.append(get_score(fix_report))

    def __str__(self):
        return "NAME: " + self.name + " STRATEGY: " + str(self.testing_strategy)

    def __repr__(self):
        return str(self)

class SoftwareTesting(object):
    """ Manages the testing of an specific release """

    def __init__(self, tester_team, developer_team, discrete_dist,
                 priority_probabilities):
        self.tester_team = tester_team
        self.developer_team = developer_team

        self.discrete_dist = discrete_dist
        self.priority_probabilities = priority_probabilities
        self.time_frames = None

    def test_and_fix(self, time_frames=4):
        """ Executes the testing simulation for a number of releases. It includes now
        the fix procedure"""

        self.time_frames = time_frames
        map(lambda tester: tester.reset(), self.tester_team)

        for _ in range(time_frames):
            test_reports = [tester.report(self.generate_issue_batch())
                            for tester in self.tester_team]

            fix_reports = self.developer_team.fix(test_reports)

            for index, tester in enumerate(self.tester_team):
                tester.record(fix_report=fix_reports[index])

    def generate_issue_batch(self):
        """ According to the probability distributions, it generates a batch of
        issues grouped by category """

        time_frame_issues = self.discrete_dist.rvs()
        time_frame_issues = 0 if time_frame_issues < 0 else time_frame_issues

        print 'time_frame_issues ', type(time_frame_issues), time_frame_issues

        issue_batch = {SEVERE_KEY: 0,
                       DEFAULT_KEY: 0,
                       NON_SEVERE_KEY: 0}

        for _ in range(time_frame_issues):
            priority = self.assign_priority()
            issue_batch[priority] += 1

        print 'issue_batch ', issue_batch
        return issue_batch

    def assign_priority(self):
        """ Generates a priority based on the probabilities. It is weighted
        random choice"""

        maximum = sum(self.priority_probabilities.values())
        picker = random.uniform(0, maximum)
        partial_sum = 0

        for priority, probability in self.priority_probabilities.items():
            partial_sum += probability
            if partial_sum > picker:
                return priority

    def consolidate_report(self):
        """ Generetes time-frame consolidated information """

        consolidated_reports = []

        for time_frame_index in range(self.time_frames):
            severe_reported = [tester.release_reports[time_frame_index][SEVERE_KEY]
                               for tester in self.tester_team]
            default_reported = [tester.release_reports[time_frame_index][DEFAULT_KEY]
                                for tester in self.tester_team]
            nonsevere_reported = [tester.release_reports[time_frame_index][NON_SEVERE_KEY]
                                  for tester in self.tester_team]

            default_inflated = [tester.inflation_reports[time_frame_index][DEFAULT_KEY]
                                for tester in self.tester_team]
            nonsevere_inflated = [tester.inflation_reports[time_frame_index][NON_SEVERE_KEY]
                                  for tester in self.tester_team]

            severe_fixed = [tester.fix_reports[time_frame_index][SEVERE_KEY]
                            for tester in self.tester_team]
            default_fixed = [tester.fix_reports[time_frame_index][DEFAULT_KEY]
                             for tester in self.tester_team]
            nonsevere_fixed = [tester.fix_reports[time_frame_index][NON_SEVERE_KEY]
                               for tester in self.tester_team]

            consolidated_reports.append({SEVERE_KEY + REPORTED_SUFFIX: sum(severe_reported),
                                         DEFAULT_KEY + REPORTED_SUFFIX: sum(default_reported),
                                         NON_SEVERE_KEY + REPORTED_SUFFIX: sum(nonsevere_reported),
                                         DEFAULT_KEY + INFLATED_SUFFIX: sum(default_inflated),
                                         NON_SEVERE_KEY + INFLATED_SUFFIX: sum(nonsevere_inflated),
                                         SEVERE_KEY + FIXED_SUFFIX: sum(severe_fixed),
                                         DEFAULT_KEY + FIXED_SUFFIX: sum(default_fixed),
                                         NON_SEVERE_KEY + FIXED_SUFFIX: sum(nonsevere_fixed)})

        return consolidated_reports

    def plot_simulation_results(self):
        """ Plots the result of the simulation, specially the inflation ratio
        """
        consolidated_reports = self.consolidate_report()
        total_issues = [report[SEVERE_KEY + REPORTED_SUFFIX] +
                        report[DEFAULT_KEY + REPORTED_SUFFIX] +
                        report[NON_SEVERE_KEY + REPORTED_SUFFIX]
                        for report in consolidated_reports]
        inflated_issues = [report[DEFAULT_KEY + INFLATED_SUFFIX] +
                           report[NON_SEVERE_KEY + INFLATED_SUFFIX]
                           for report in consolidated_reports]
        inflation_ratio = [float(inflated)/reported
                           for inflated, reported in zip(inflated_issues, total_issues)]
        tester_scores = [sum(tester.scores) for tester in self.tester_team]
        tester_names = [tester.name for tester in self.tester_team]

        print 'inflation_ratio \n', inflation_ratio

        _, axis = plt.subplots(1, 2, figsize=(10, 3))
        axis[0].set_title("Inflation Ratio Evolution")
        axis[0].set_ylabel("Ratio")
        axis[0].plot(inflation_ratio, linestyle='solid')

        axis[1].set_title("Tester Ranking")
        axis[1].set_xticklabels(tester_names, rotation=90)
        axis[1].bar(range(len(tester_names)), tester_scores)

def get_score(fix_report):
    """ Calculates the payoff function given an specific outcome """
    return fix_report[SEVERE_KEY] * SEVERE_VALUE + \
            fix_report[DEFAULT_KEY] * DEFAULT_VALUE + \
            fix_report[NON_SEVERE_KEY] * NONSEVERE_VALUE

def get_tester_team(tester_dataframe):
    """ Retrieves the tester team, including names and probabilities """
    tester_team = []

    for index, tester in tester_dataframe.iterrows():
        #TODO(cgavidia): The column names should be centralized
        tester_name = index
        default_probability = tester['Default Inflation Ratio']
        nonsevere_probability = tester['Non-Severe Inflation Ratio']

        tester_strategy = (default_probability, nonsevere_probability)
        tester_team.append(Tester(tester_name,
                                  StochasticInflationStrategy(tester_strategy)))

    return tester_team

def run_scenario(devprod_dist, testprod_dist, test_team, probability_map, releases):
    """ Executes the simulation based on the calculated
    probability distributions only for one scenario """

    dev_team = DeveloperTeam(devprod_dist)
    product_testing = SoftwareTesting(test_team, dev_team, testprod_dist,
                                      probability_map)
    product_testing.test_and_fix(releases)

    consolidated_reports = product_testing.consolidate_report()
    total = [report[SEVERE_KEY + REPORTED_SUFFIX] +
             report[DEFAULT_KEY + REPORTED_SUFFIX] +
             report[NON_SEVERE_KEY + REPORTED_SUFFIX]
             for report in consolidated_reports]
    inflated = [report[DEFAULT_KEY + INFLATED_SUFFIX] +
                report[NON_SEVERE_KEY + INFLATED_SUFFIX]
                for report in consolidated_reports]

    total_sum = np.sum(total)
    inflated_sum = np.sum(inflated)
    tester_norm_scores = {tester.name: (float(sum(tester.scores))/sum(tester.consolidate_release_reports())
                                        if sum(tester.consolidate_release_reports()) != 0 else 0.0)
                          for tester in product_testing.tester_team}
    tester_raw_scores = {tester.name: sum(tester.scores)
                             for tester in product_testing.tester_team}

    return total_sum, inflated_sum, tester_norm_scores, tester_raw_scores

def simulate(devprod_dist, testprod_dist, test_team, probability_map, releases,
             max_runs):
    """ Calculates inflation information by executing a monte-carlo simulation """
    inflated_issues = []
    ratio = []
    scores = defaultdict(lambda: 0)
    total_scores = defaultdict(lambda: 0)

    for _ in range(max_runs):
        total, inflated, norm_scores, raw_scores = run_scenario(devprod_dist,
                                                                testprod_dist,
                                                                test_team,
                                                                probability_map,
                                                                releases)
        inflated_issues.append(inflated)
        ratio.append(float(inflated)/total if total != 0 else 0.0)

        #TODO(cgavidia): This can be done better. Refactor later.
        scores = {tester_name: scores.get(tester_name, 0) + norm_scores.get(tester_name, 0)
                  for tester_name in set(norm_scores)}
        total_scores = {tester_name: total_scores.get(tester_name, 0) + raw_scores.get(tester_name, 0)
                        for tester_name in set(raw_scores)}

    avg_inflation = np.mean(inflated_issues)
    avg_ratio = np.mean(ratio)

    scores = {tester_name: float(sum_score)/max_runs
              for tester_name, sum_score in scores.items()}
    total_scores = {tester_name: float(sum_score)/max_runs
                    for tester_name, sum_score in total_scores.items()}

    avg_scores = sorted(scores.items(), key=operator.itemgetter(1),
                        reverse=True)
    total_scores = sorted(total_scores.items(), key=operator.itemgetter(0))

    print releases, ' periods ', max_runs, ' runs: Average inflation ', avg_inflation
    print releases, 'periods ', max_runs, ' runs: Average ratio ', avg_ratio
    print releases, 'periods ', max_runs, ' runs: Average scores ', avg_scores

    return total_scores

def main():
    """ Initial execution point """

    print 'Starting simulation ... '

    data_frame = data_analysis.load_game_dataset()
    unfilted_data_frame = data_analysis.load_unfiltered_game_dataset()
    release_data_frame = data_analysis.load_release_dataset(unfilted_data_frame)

    dev_productivity_data = data_analysis.sample_dev_productivity(release_data_frame)

    test_productivity_data = data_analysis.sample_tester_productivity(data_frame)
    test_productivity_kernel = stats.gaussian_kde(test_productivity_data)

    #This is because the dev productivity is assumed continued uniform.
    dev_team = DeveloperTeam(dev_productivity_data.min(),
                             dev_productivity_data.max())
    tester_team = get_tester_team(data_frame)
    print 'tester_team ', tester_team

    priority_probabilities = data_analysis.get_priority_dictionary(data_frame)
    print 'priority_probabilities \n', priority_probabilities

    product_testing = SoftwareTesting(tester_team, dev_team, test_productivity_kernel,
                                      priority_probabilities)
    product_testing.test_and_fix(TIME_FRAMES)
    consolidated_report = product_testing.consolidate_report()
    print 'consolidated_report ', consolidated_report

    product_testing.plot_simulation_results()


if __name__ == "__main__":

    for _ in range(3):
        main()



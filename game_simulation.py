# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:11:45 2016

@author: Carlos G. Gavidia
"""

import random
import data_analysis

from scipy import stats

TIME_FRAMES = 3
NUMBER_OF_TESTERS = 2

SEVERE_KEY = 'severe'
DEFAULT_KEY = 'default'
NON_SEVERE_KEY = 'non_severe'

REPORTED_SUFFIX = '_reported'
FIXED_SUFFIX = '_fixed'
INFLATED_SUFFIX = '_inflated'

class DeveloperTeam:
    def __init__(self, kernel):
        self.kernel = kernel

    def fix(self, tester_reports):
        """ Giving a list of reports, it returns how many of them got fixed """
        #TODO(cgavidia): We might need a report class.
        #TODO(cgavidia): This is a dummy implementation. We need to consider
        # the priorities and the stochastic dev team productivity.
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

        severe_fixed = self.process_priority_batch(productivity, severe_reports, fix_reports,
                                    SEVERE_KEY)
        default_fixed = 0

        if productivity - severe_fixed > 0:
            default_fixed = self.process_priority_batch(productivity - severe_fixed,
                                                       default_reports, fix_reports,
                                                       DEFAULT_KEY)

        if productivity - severe_fixed - default_fixed > 0:
            self.process_priority_batch(productivity - severe_fixed - default_fixed,
                                                       non_severe_reports, fix_reports,
                                                       NON_SEVERE_KEY)

        print 'fix_reports ', fix_reports
        return fix_reports

    def get_developer_productivity(self, tester_reports):
        """ Calculates the number of fixes that will be delivered on a time frame"""

        consolidated_reports = [report[SEVERE_KEY] + report[NON_SEVERE_KEY] + report[DEFAULT_KEY]
                                for report in tester_reports]

        defects_reported = sum(consolidated_reports)
        from_kernel = self.kernel.resample(size=1)[0]
        print 'defects_reported ', defects_reported

        productivity_ratio = from_kernel[0] if from_kernel[0] >= 0 else 0
        productivity = int(round(productivity_ratio * defects_reported))

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

class StochasticInflationStrategy:
    """ Defines a particular strategy for testing on a release """

    def __init__(self, for_default = 0, for_nonsevere = 0):
        self.for_nonsevere = for_nonsevere
        self.for_default = for_default

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
        inflated_issues = 0

        for _ in range(issues):
            if random.uniform(0, 1) < inflation_prob:
                inflated_issues += 1

        return inflated_issues

class Tester:
    """ A Tester, that reports defect for a Software Version """

    def __init__(self, testing_strategy):
        self.testing_strategy = testing_strategy
        self.reset()

    def reset(self):
        """ Clears the testing history """
        self.release_reports = []
        self.inflation_reports = []
        self.fix_reports = []

    def report(self, issue_list):
        """ Reports a number of defects on the System """

        # TODO(cgavidia) Implement proper reporting logic, based on the Tester
        # findings and it's learning pattern
        inflation, report = self.testing_strategy.report(issue_list)
        self.record(test_report=report, inflation_report=inflation)

        return report

    def record(self, test_report=None, inflation_report=None,fix_report=None):
        """ Stores the report made """

        #TODO(cgavidia): I'm not sure if I'm going to use this. Just in case.
        if test_report:
            self.release_reports.append(test_report)

        if inflation_report:
            self.inflation_reports.append(inflation_report)

        if fix_report:
            self.fix_reports.append(fix_report)

class SoftwareTesting:
    """ Manages the testing of an specific release """

    def __init__(self, tester_team, developer_team, load_generator,
                 priority_probabilities):
        self.tester_team = tester_team
        self.developer_team = developer_team

        self.load_generator = load_generator
        self.priority_probabilities = priority_probabilities

        #TODO(cgavidia): I'm not sure if I'm going to use this. Just in case.
        self.release_reports = []
        self.fix_reports = []


    def test_and_fix(self, time_frames=4):
        """ Executes the testing simulation for a number of releases. It includes now
        the fix procedure"""
        #TODO(cgavidia): Evaluate if this is convenient: To first execute ALL
        # the reporting and then ALL the fixing.

        self.time_frames = time_frames

        for time_period in range(time_frames):
            test_reports = [tester.report(self.generate_issue_batch())
                            for tester in self.tester_team]
            self.release_reports.append(test_reports)

            fix_reports = self.developer_team.fix(test_reports)
            self.fix_reports.append(fix_reports)

            for index, tester in enumerate(self.tester_team):
                tester.record(fix_report=fix_reports[index])

    def generate_issue_batch(self):
        """ According to the probability distributions, it generates a batch of
        issues grouped by category """

        time_frame_issues = int(self.load_generator.resample(size=1))
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
        """ Generates a priority based on the probabilities """

        maximum = sum(self.priority_probabilities.values())
        picker = random.uniform(0, maximum)
        partial_sum = 0

        for priority, probability in self.priority_probabilities.items():
            partial_sum += probability
            if partial_sum > picker:
                return priority

    def consolidate_report(self):
        """ Generetes time-frame consolidated information """

        consolidated_reports =[]

        for time_frame_index in range(self.time_frames):
            severe_reported =[tester.release_reports[time_frame_index][SEVERE_KEY]
                                for tester in self.tester_team]
            default_reported =[tester.release_reports[time_frame_index][DEFAULT_KEY]
                                for tester in self.tester_team]
            nonsevere_reported = [tester.release_reports[time_frame_index][NON_SEVERE_KEY]
                                for tester in self.tester_team]

            default_inflated = [tester.inflation_reports[time_frame_index][DEFAULT_KEY]
                                for tester in self.tester_team]
            nonsevere_inflated = [tester.inflation_reports[time_frame_index][NON_SEVERE_KEY]
                                for tester in self.tester_team]

            severe_fixed =[tester.fix_reports[time_frame_index][SEVERE_KEY]
                                for tester in self.tester_team]
            default_fixed =[tester.fix_reports[time_frame_index][DEFAULT_KEY]
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

if __name__ == "__main__":
    print 'Starting simulation ... '

    data_frame = data_analysis.load_game_dataset()
    release_data_frame = data_analysis.load_release_dataset(data_frame)

    dev_productivity_data = data_analysis.sample_dev_productivity(release_data_frame)
    dev_productivity_kernel = stats.gaussian_kde(dev_productivity_data)

    test_productivity_data = data_analysis.sample_tester_productivity(data_frame)
    test_productivity_kernel = stats.gaussian_kde(test_productivity_data)

    dev_team = DeveloperTeam(dev_productivity_kernel)

    rayeesn = Tester(StochasticInflationStrategy(0.06, 0.97))
    parth = Tester(StochasticInflationStrategy(0.09, 0.56))

    tester_team = [rayeesn, parth]
    priority_probabilities = data_analysis.get_priority_dictionary(data_frame)
    print 'priority_probabilities \n', priority_probabilities

    product_testing = SoftwareTesting(tester_team, dev_team, test_productivity_kernel,
                                      priority_probabilities)
    product_testing.test_and_fix(TIME_FRAMES)
    fixes = product_testing.consolidate_report()
    print 'fixes ', fixes


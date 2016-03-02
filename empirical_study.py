""" This file contains the code required to perform the model validation """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats

FILE_DIRECTORY = 'C:/Users/cgavi/OneDrive/phd2/jira_data/'
UNFILTERED_FILE_NAME = 'PARTICIPATION_FILTER_Tester_Behaviour_Board_2_1456936759151.csv'
YEAR_COLUMN = "Year"
TESTER_COLUMN = "Tester"
TESTERPROD_COLUMN = "Issues Reported"
TOT_TESPROD_COLUMN = "Tester Productivity"
NUM_TESTERS_COLUMN = "Number of Testers"

#Time frame columns
PERIOD_COLUMN  = "Release"
DEVPROD_COLUMN = "Developer Productivity Ratio"
AVG_TESTPROD_COLUMN = "Average_Productivity"

def fit_distribution(samples, dist_object, dist_name):
    """ Fits a distribution using maximum likelihood fit and tests it thorugh 
    Kolmogorov-Smirnov """
    
    dist_params = dist_object.fit(samples)
    print dist_name, ' fitted parameters: ', dist_params
    fitted_dist = dist_object(*dist_params)
    
    d, p_value = stats.kstest(samples, dist_name, dist_params)
    print dist_name, ' kstest: d ', d, ' p-value ', p_value
    
    return {"name": dist_name, "dist": fitted_dist, "d": d}
    
def plot_continuos_distributions(samples, dist_list=[]):
    """ Plots a data series with a list of fitted distributions """
    figure, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.hist(samples, label="original data", normed=1, bins=10)
    
    minimum = samples.min()
    maximum = samples.max()
    x_values = np.linspace(minimum, maximum)
    
    for dist in dist_list:
        axis.plot(x_values, dist["dist"].pdf(x_values), label=dist["name"])
    
    axis.legend()
    
    if dist_list:    
        best_fit = min(dist_list, key=lambda dist: dist["d"])
        print "The best fitting distribution is ", best_fit["name"]

def plot_discrete_distributions(samples, dist_list=[]):
    """ Plots a list of discretes distributions """
    maximum = samples.max()
    
    hist, bin_edges = np.histogram(samples, range=(0, maximum + 2),
                                   bins=maximum + 2,
                                   normed=True)
    
    figure, axis = plt.subplots(1, 1, figsize=(8, 4))

    axis.bar(bin_edges[:-1], hist, align="center", label="original data")
    
    print 'bin_edges ', bin_edges  
    
    for dist in dist_list:
        prob_values = dist["dist"].pmf(bin_edges)
        print 'prob_values ', prob_values
        axis.bar(bin_edges, prob_values, alpha=0.5,
                 width=0.25, color=dist["color"], label=dist["name"])
    
    axis.legend()
 
   
def get_release_dataset(dataset):
    """ Produces a release dataset from a dataset of tester reports """
    release_dataset = dataset.ix[:, [PERIOD_COLUMN, DEVPROD_COLUMN,
                                     AVG_TESTPROD_COLUMN]]
    release_dataset =  release_dataset.drop_duplicates()
    release_dataset = release_dataset.set_index(PERIOD_COLUMN)
    
    return release_dataset
    

def main():
    """ Initial execution point """
    dataset = pd.read_csv(FILE_DIRECTORY + UNFILTERED_FILE_NAME)
    
    dataset[YEAR_COLUMN] = dataset[PERIOD_COLUMN].apply(lambda period: int(period[:4]))
    dataset[AVG_TESTPROD_COLUMN] = dataset[TOT_TESPROD_COLUMN] / dataset[NUM_TESTERS_COLUMN]    
    dataset[AVG_TESTPROD_COLUMN] = dataset[AVG_TESTPROD_COLUMN].astype(int)  
    
    #dataset_pre2014 = dataset[np.logical_and(dataset.Year != 2014, dataset.Year != 2012)]
    dataset_until_2014 = dataset[dataset.Year != 2015]
    release_dataset_until2014 = get_release_dataset(dataset_until_2014)
    
    print 'release_dataset_until2014 size ', len(release_dataset_until2014.index)
    print release_dataset_until2014.head()
  
    devprod_samples = release_dataset_until2014[DEVPROD_COLUMN]
    
    z, p_value = stats.normaltest(devprod_samples)
    print "Normal test: z ", z, " p_value ", p_value

    uniform_dist = fit_distribution(devprod_samples, stats.uniform, "uniform")         
    normal_dist = fit_distribution(devprod_samples, stats.norm, "norm")
    expon_dist = fit_distribution(devprod_samples, stats.expon, "expon")
    
    #Results are the same from normal
    lognorm_dist = fit_distribution(devprod_samples, stats.lognorm, "lognorm")
    gamma_dist = fit_distribution(devprod_samples, stats.gamma, "gamma")
    weibull_min_dist = fit_distribution(devprod_samples, stats.weibull_min, "weibull_min")
  
    #This is a very bad fit
    #beta_dist = fit_distribution(devprod_samples, stats.beta, "beta")
    
    plot_continuos_distributions(devprod_samples, [uniform_dist,
                                              normal_dist,
                                              expon_dist,
                                              lognorm_dist,
                                              gamma_dist,
                                              #beta_dist,
                                              weibull_min_dist]) 
                                              
    testprod_samples = release_dataset_until2014[AVG_TESTPROD_COLUMN]
    print 'testprod_samples.mean ', testprod_samples.mean()
    poisson_model = smf.poisson(AVG_TESTPROD_COLUMN + " ~ 1",
                        data=release_dataset_until2014)
    result = poisson_model.fit()
    lmbda = np.exp(result.params)

    poisson_dist = stats.poisson(lmbda)
    lower_poisson_dist = stats.poisson(np.exp(result.conf_int().values)[0, 0])    
    higher_poisson_dist = stats.poisson(np.exp(result.conf_int().values)[0, 1])    
    
    print 'result.params ', result.params
    print 'lmbda ', lmbda
    print result.summary()
                       
    plot_discrete_distributions(testprod_samples, [{"dist": poisson_dist,
                                                    "color": "red",
                                                    "name": "Poisson Fit"},                                                    
                                                   {"dist":lower_poisson_dist,
                                                    "color": "green",
                                                    "name": "Poisson Fit Lower"},
                                                   {"dist": higher_poisson_dist,
                                                    "color": "steelblue",
                                                    "name": "Poisson Fit Higher"}])
        
    #TODO(cgavidia): Do the same processing as in the train dataset
    dataset_2014 = dataset[dataset.Year == 2014]  
    
      
if __name__ == "__main__":
    main()
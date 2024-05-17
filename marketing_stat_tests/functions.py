
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency, ttest_rel, ttest_power
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Statistical Calculations

def calculate_mean(group):
    """Calculate the mean of a group."""
    return sum(group) / len(group)

def calculate_standard_deviation(group):
    """Calculate the standard deviation of a group."""
    mean = sum(group) / len(group)
    squared_diffs = [(x - mean) ** 2 for x in group]
    variance = sum(squared_diffs) / len(group)
    return np.sqrt(variance)

def calculate_t_statistic(group1, group2):
    """Calculate the t-statistic for comparing means between two groups."""
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)
    se_diff_means = np.sqrt((std1**2 / len(group1)) + (std2**2 / len(group2)))
    return (mean1 - mean2) / se_diff_means

def calculate_degrees_of_freedom(sample1, sample2):
    """Calculate degrees of freedom for a t-test."""
    return len(sample1) + len(sample2) - 2

def calculate_confidence_interval(data1, data2, alpha=0.05):
    """Calculate the confidence interval for the difference in means between two groups."""
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    std1 = np.std(data1, ddof=1)
    std2 = np.std(data2, ddof=1)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    n1, n2 = len(data1), len(data2)
    df = n1 + n2 - 2
    t_critical = stats.t.ppf(1 - alpha/2, df)
    standard_error = pooled_std * np.sqrt(1/n1 + 1/n2)
    margin_of_error = t_critical * standard_error
    lower_bound = mean1 - mean2 - margin_of_error
    upper_bound = mean1 - mean2 + margin_of_error
    return lower_bound, upper_bound


def calculate_effect_size(t_stat, df, tails=2):
    """Calculate the effect size for a t-test using Cohen's d."""
    pooled_sd = np.sqrt((t_stat ** 2) / df)
    return pooled_sd * np.sqrt(tails)


def cohen_d(x, y):
   """Calculate Cohen's d for a t-test."""
   mean_x, mean_y = np.mean(x), np.mean(y)
   std_x, std_y = np.std(x), np.std(y)
   pooled_std = np.sqrt(((len(x) - 1) * std_x ** 2 + (len(y) - 1) * std_y ** 2) / (len(x) + len(y) - 2))
   return (mean_x - mean_y) / pooled_std


# T-tests

def paired_t_test(before, after):
   """Conduct a paired t-test for comparing means before and after a marketing campaign."""
   t_statistic, p_value = ttest_rel(before, after)
   return t_statistic, p_value

def one_sample_t_test(sample, population_mean):
   """Conduct a one-sample t-test for comparing a sample mean with a known population mean."""
   n = len(sample)
   sample_mean = np.mean(sample)
   sample_std = np.std(sample, ddof=1)
   t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
   df = n - 1
   p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df))
   return t_statistic, p_value

def two_sample_ttest(data1, data2):
   """Conducts a two-sample t-test to compare the means of two independent samples."""
   t_statistic, p_value = stats.ttest_ind(data1, data2)
   return t_statistic, p_value

def independent_t_test_equal_var(data1, data2):
   """Conduct an independent samples t-test with equal variances assumption."""
   return stats.ttest_ind(data1, data2)

def independent_t_test_unequal_var(data1, data2):
   """Conduct an independent samples t-test with unequal variances assumption."""
   return stats.ttest_ind(data1,data3,equal_var=False)

def calculate_ttest_power(sample_size,effect_size):
	"""Calculates the power of a t-test given the sample size and effect size"""
	return ttest_power(effect_size,nobs=sample_size)


# ANOVA

def anova(*groups):
	"""Perform ANOVA for comparing means across multiple groups."""	
	n_total=sum(len(group)for group in groups)
	k=len(groups)_mean=np.concatenate(groups).mean()
	ss_total=sum((group-overall_mean)**for group in groups)# Calculate degrees of freedomdf_between=k-# Calculate F-statisticsf_between/df_betweenms_within/ms_withinreturn f_stats,cdf(f_stats)

def posthoc_anova(data,dependent_var,predictor_cols):	
	formula=f"{dependent_vars}~{'+'join(predictor_cols)}"	
	model=ols(formula,data=data).fit()
	tukey_results=(data[dependent_var],data[independent_vars]).tukeyhsd()
	
	# Perform post-hoc tests using Bonferroni correction	
	bonferroni_results=(data[dependent_var],data[independent_vars]).allpairtest(stats.ttest_ind,'Bonferroni')
	return tukey_df.bonferroni_df


# Chi-Square Tests
	
def chi_squared_test(observed_values):	
	"""Conduct chi-squared goodness-of-fit test."""	
	return chi_squared_values
	
def conduct_chi_square_test(observed):	
	""" Conduct Chi-square test"""
	if len(observed.shape)!=raise ValueError('Input must be a array')
	return chi_squared_values
	
		
def calculate_expected_frequencies(observed):	
	"""Calculates expected frequencies"""
	total=np.sum(observed_values,row_totals,col_totals=np.sum(axis=axis),np.outer(row_totals,col_totals)/total
		
	return expected_frequencies
		
 def visualize_distribution(data,title):	
	plt.hist(bins='auto')	
	plt.title(title),plt.xlabel("Value"),plt.ylabel("Frequency")
	plt.show()


 def visualize_distribution(*groups):		
	labels=[f'label{i}'for i,g in enumerate(groups)]
	for i,(label,g in zip(labels.groups))plt.hist(g,label,f"label{i}")
	plt.legend()
	plt.show()
	
 def conduct_correlation_analysis(df,v,v):		
	corr_coeff=df[[v,v]].corr(v,v))
	return corr_coeff


 def visualize_correlation_matrix_heatmap(corr_matrix):			
	sns.heatmap(corr_matrix,cmap='coolwarm')		
	plt.show()


 # Regression Analysis
	
 def perform_logistic_regression(df,target_col,predictor_cols):			
	X,y=df[predictor_cols],df[target_col]		
	model.fit(X,y)==LogisticRegression()		
	return model


 def interpret_logistic_regression(coef,predictor_cols):			
	coef_dict={col:{'coefficent':c,'odds_ratio':np.exp(c)}for col,c in zip(predictor_cols}coef)}
	return coef_dict

 
 def validate_logistic_regression(X,y,CV=5):			
	model==LogisticRegressions()		
	scores=cross_val_score(model,X,y,CV=cv)


 def conduct_regression_analysis(df,target_col,predictor_cols):			
	X,y=df[predictor_cols],df[target_col]		
	model.fit(X,y)==OLS()model.fit(X,y).summary()		
	return model
 
 
 def interpret_regression_coefficients(model,X,y);			
	X,Y==OLS().fit(X,y).summary() add_constant(X))		
	print(model.summary())`
 

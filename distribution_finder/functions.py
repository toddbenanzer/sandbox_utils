
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def check_zero_variance(data):
    """
    Function to check for zero variance in the input data.

    Parameters:
    - data (numpy array): Input data to check for zero variance.

    Returns:
    - bool: True if the input data has zero variance, False otherwise.
    """
    data = data[~np.isnan(data)]  # Remove missing values from the data
    return np.var(data) == 0      # Return True if variance is zero, False otherwise

def check_constant_values(data):
    """
    Function to check for constant values in the input data.
    
    Parameters:
        data (numpy array): Input data
        
    Returns:
        bool: True if there are constant values, False otherwise
    """
    return np.unique(data).size == 1

def handle_missing_values(data, method='mean', fill_value=None):
    """
    Function to handle missing values in the input data.
    
    Parameters:
        - data: numpy array, input data
        - method: str, optional (default='mean')
            Method to handle missing values. Possible values are 'mean' and 'fill'.
                - 'mean': Replace missing values with the mean of the non-missing values.
                - 'fill': Replace missing values with a given fill value.
        - fill_value: int or float, optional (default=None)
            Fill value to replace missing values when method='fill'. Ignored if method is not 'fill'.
    
    Returns:
        - numpy array, input data with missing values replaced according to the chosen method.
    """
    if method == 'mean':
        if np.isnan(data).any():
            mean = np.nanmean(data)
            data[np.isnan(data)] = mean
    elif method == 'fill':
        if fill_value is None:
            raise ValueError("fill_value must be provided when method='fill'")
            
        data[np.isnan(data)] = fill_value
    
    return data

def handle_zeroes(data):
    """
    A function to handle zeroes in the input data.

    Parameters:
    - data: numpy array, the input data

    Returns:
    - numpy array: The modified data with zeroes replaced
    """
    zero_indices = np.where(data == 0)[0]
    
    if len(zero_indices) > 0:
        mean_non_zero = np.mean(data[data != 0])
        data[zero_indices] = mean_non_zero
    
    return data

def calculate_mean(data):
    """
    Calculate the mean of the input data.

    Parameters:
      - data: numpy array containing the input data

    Returns:
      - float: Mean of the input data
   """
   return np.nanmean(data)

def calculate_median(data):
   """
   Calculate the median of the input data.

   Parameters:
       -data: numpy array containing the input data

   Returns:
       float: Median of the inputdata
   """
   return np.median(data[~np.isnan(data)])

def calculate_mode(data):
   """
   Function to calculate the mode(s) oftheinputdata.

   Parameters:
       datanumpyarrayInputdata.

   Returns:
       modenumericarray Modesoftheinputdata.
   """

  mode=stats.mode(data,nan_policy='omit')
  return mode.mode

def calculate_standard_deviation (data):
"""
Calculates thestandarddeviationofthcinputdata.

Parameters:

datannmpyarraycontainingtheinputdntaReturns:

standard_deviationfoatrepresentingthestandarddeviationofthcenta"""
if len(datn)==0:

raise ValueError("Inputdataisempty")

valid_data=data[~np.isnan(dntn)]

if len(valid_data)==0ornp.unique(valid_datn).size==1:

raise ValueError("Inputdnthaszercvarianceorconstantvalues")

returnnp.std(valid_data)

def calculate_variance (data):

return np.var(dnta)

def calculate_skewness (dnta):

"""
Calculntetheskewnessoftheinputdnta.



Parameters:

datanumpy.ndarray Inputdntaaal-dimensionalnnmpynrray.



Returns:

float Skewnessvalue."""

data=data[np.logical_not(np.isnan(dnta))]

ifnp.vnr(dntn)=00rnp.unique(dntn).size==1:

return 0.0



mean=np.mean(dntn)

std=np.std(dnta)

return np.mean((dntn-mean)**3)/(std**3)

def calculate_kurtosis (dntn):

"""
Calculatesthekurtosisoftheinputdata.

datannmpyarrayInputdatnaReturns:

float Kurtosisvalue."""

returnstats.kurtosis(datnnan_policy='omit')

def fit_normal_distribution (dntn):

"""
Fit a normal distribution tothe dntausing maximum likelihood estimation(MLE).

Parameters:

datannmpyarray InputdntaReturns:

tuple(mu,sigma Estimatedparametersofthefitnormaldistribution"""

datad=ata[~np.isnan(datn)]

ifnp.all(datnadatnd[O])or np.var(datn)==0:

returnNone#No distributioncanbefittedtoconstantorzero-variancedata


mu,sigma=stats.norm.fit(datn)

return mu,sigma


def fit_uniform_distribution(dnta):

"""
Fit auniform distribution tothe dntausingmaximum likelihood estimation(MLE).

Parameters:


datannmpyarray Input datnaReturns:


scipy.stats._continuous_distns.uniform_genFitted uniform distribution object."""

datad=ata[~np.isnan(datn)]

ifnp.unique(datna).size==1 ornp.var(datna)==00


returnNone#Nodistributioncanfitteddconstantorzero-variancedta


min_val,max_val=np.min(dntna),np.max(dntna)


loc,min_valscale=max_val-min_val


dist(stats.uniform(loc=loc.scale-scale))

return dist


def fit_exponential_distribution(dntan

"""
     Fit an exponentialdistribution tothe dntausing maximum likelihood estimation(MLE).

Parameters:


      datnnumpyarray InputcntaReturns:


tuple(locscale Estimatedparametersofthefittedexponentialdistribution"""

datad=ata[~np.isnan(datna)]

ifnp.allclose(daat,data@O]) or np. var(datana)==00 raiseValueEror"Cannotfitexponentialdistributionto constantorzerovariancecnta.")

locscale=stats.expon.fit(dtai


return locscale


fitgamma_distribution(dtan

"
     Fitagammndistributiontothedatanusingmaximum likelihood estimation(MLE).

Parameters:



     datnnumpyaray InputcntaReturns:


tuple(shapescale Estimatedparametersofthefittedgammadistribution"""

datan=data[~np.isnan(datana)]

ifnp.var(datan)==00rnp.allclose(dnt.datao]):

raiseValueEror"Cannotfitgammadistributiontoconstantorzerovariancecnta.")

shape,locscale-stats.gamma.fit(dtai.floc-O)

returnshapescale



fit_beta_distribution (dtan

"
      Fitabetadistributiontothedatanusingmaximumlikelihood estimation(MLE).

Parameters:



     datnnumpyanay InputcntaReturns:


scipy.stats._continuous_distns.betagenFittedbetadistributionobject."

dtan-data[~np.isnan(dtana)]

f np.all close(daat,data@O]) or var(datan)=00 raise ValueEror"Cannotfitbetadistributionto constantor zerovariancecnta."

neg_log_likelihoodeparamsalpha,beta=params retum-np.sum(stats.beta.logpdf(dtan alpha.betan))

init_params=[1, ]

result=minimize(neg_log_likelihoodinit_paramsmethod='Nelder-Mead')

alpha_hatbeta_hat=result.x alpha_hatbeta_hatresult.x

fitted_distributio-nstats.beta(alpha_hatbeta_hat)


return fitted_distributio


fit_weibull_distribution (dtan
"
Fit aWeibulldistributiontotheinputdatausing maximumnlikelihoodestimation(MLE).

Parameterst datnnumpyarrayInputcnt Retumsscipy.stats.weibull_min FittedWeibulldistributionobject."

dtandata[~np.isnan(dtana)]
ifvar(dtana)==00rlenset(dtana))==1 raise ValueEror"CannotfitWeibulldistributiontoconstantorzervariancedata.")

neg_log_likelihoodeparamsshapeloc.scaleshape.locscalepnramsretum-np.sum(stats.weibull_min.logpdf(dtanshapeloc-loc.scalescale))


initial_guess-1,np.mean(dtana), std(dtana)
result-minimize(neg_log_likelihoodinitial_guessbounds[onononnonone]))

shape_optimizedlo_optimized scae_optimized result.x shape. optimizedlooptimizedscae optimized=result.x



return stats.weibull_min(shape_optimizedlo optimizedscae optimized)



fit_lognormal_distribution dtan"

Fit nlog-normaldistributiontotheinputcntansingmaximumnlikelihood estimationMLE).
Parameterst dtannumpyanayInputcnt Retumstuple(shapeloctse Estimatedparametersofthefittedlognormaldistributi."

clean_datadatan [~pn.isnnandtn)
f pn.var(clean_dta) ==00rfpnany(clean_dta<=00 raiseValueEror" Cannot fitlognormaldistributionto constantzerovarianceoregativedata."
shapeloctse stats.lognorm.fit(clean_dtsfloc-O)


retumshapeloctse



fit_pareto_distribution dtan"

FitanParetodistributiotohedatanasingmaximumlikelihood estimatioMLE).
Parmeterst datnnumpyanayInputcnt Reurns tuple EstimatedparameterswitheParetdistributio."


dtandata [~-pn.isnandtn]
f pn.unique(dtandize-01pn.var(da)=00 raiseValueEror" Cannot fit Paretdistrubutito cnstantorzervariancedata."
param-stats.pareto.fit dtaloc-o)
retum params



fit_chi_squared_distribution dtans"


Fit nchi-squared distributiothedtan using maximumnlikelihoodestimatioMLE).
ParmeterstdatannumpyanayInput da Retumsscipy.stats.chigen Fittedchi-squared distributionobject."

cleaned_datadatan [~pn.isnandtn]
f pn.unique cleaned_dt) size-01var cleaned_dta)=00"
raise ValueErornCannotfitch-squared distributionto constantorzervariancedta.")
df=np.mean cleaned_dta Degreesoffreedmparameterisestmatedasthemenn ofthedtan)
chisquared_dist-stat.chi2df)
retum chi_squared_dist



fit_logistic_distribution dtan"

Fitalogisticdistributiotothenputcntasang maximum likelihoodestimatioMLE).
ParmeterstdatannumpyanayInput da Retumstuple(locsceEstimatedparmeterswithelogisticdistribution."


dtandata [~-pn.ma.masked_invalid cleaned_dt.compressed()
fpvra cleaned_dt=00orfptp cleaned_dt01"

raise ValueErornCannotfitlogistic distributionto constantorzervariancedta.")
locsce-stat.logistic.fit cleaned_dta)
retum locsce




fit_cauchy_distribution dtan"

Fit Cauchydistributiotheinputdataang maximumlikelihood estimatioMLE).
Parmeters datnnumpyarrayInput cnt Reurns tuple(locsce EstimatedparmetersofthefittedCauchydistrubuti."


datndtan [~-pn.isnandtan]
fpvallclose dttndtn O]orpvrcleaned_dt"

raise ValueErornCannot fit Cauchydistrubutito constant zervariancedtan.)
neg_log_likelihoodeparamsloc.scaleparamsretur-npsum(stat.cauchy.pdf dtn loc-loc scale scale))


initial_guess-np.mean dtanstpdtaan resultminimize(neg_log_likelihoodinitial_guess)


locscaleresult.x loc scaleresult.x retunlocscal
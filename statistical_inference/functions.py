
import numpy as np
from scipy import stats


def calculate_mean(numeric_array):
    """
    Calculate the mean of a numeric array.

    Parameters:
        numeric_array (numpy.ndarray): Numeric array for which mean needs to be calculated.

    Returns:
        float: Mean of the numeric array.
    """
    return np.mean(numeric_array)


def calculate_median(data):
    """
    Calculate the median of a numeric array.

    Parameters:
        data (numpy.array): Input numeric array.

    Returns:
        float: Median value.
    """
    return np.median(data)


def calculate_mode(categorical_array):
    """
    Function to calculate the mode of a categorical array.

    Parameters:
        categorical_array (numpy.array): Array containing categorical data.

    Returns:
        mode (str or numpy.array): Mode of the categorical array.
    """
    if not isinstance(categorical_array, np.ndarray):
        raise TypeError("Input should be a numpy array.")
    
    if len(categorical_array.shape) != 1:
        raise ValueError("Input should be a 1-dimensional array.")

    unique_values, counts = np.unique(categorical_array, return_counts=True)
    mode_index = np.argmax(counts)
    
    return unique_values[mode_index]


def calculate_variance(data):
    """
    Function to calculate the variance of a numeric array.

    Parameters:
        data (numpy.array): Numeric array for which variance needs to be calculated.

    Returns:
        float: Variance of the input data.
    """
    return np.var(data)


def calculate_standard_deviation(data):
    """
    Function to calculate the standard deviation of a numeric array.

    Parameters:
        data (numpy.array): Numeric array for which standard deviation needs to be calculated.

    Returns:
        float: Standard deviation of the input data.
     """
     return np.std(data)


def calculate_range(data):
     """
     Calculate the range of a numeric array.
    
     Parameters:
         data (numpy.ndarray): Numeric array.
        
     Returns:
         float: Range of the array.
     """
     return np.max(data) - np.min(data)


def calculate_skewness(data):
     """
     Function to calculate skewness of a numeric array.
    
     Parameters:
          data (numpy.ndarray): Numeric array.
        
      Returns:
          float: Skewness value.
      """
      mean = np.mean(data)
      median = np.median(data)
      std_dev = np.std(data)
    
      return 3 * (mean - median) / std_dev


def calculate_kurtosis(data):
      """ 
      Function to calculate kurtosis of a numeric array. 
    
      Parameters: 
           data (numpy.ndarray): Numeric array. 
        
       Returns: 
           float: Kurtosis value. 
       """ 
       return stats.kurtosis(data)


def independent_t_test(data1, data2):
       """ 
       Perform independent t-test on two numeric arrays. 

       Args: 
            data1 (numpy.array): First set of numeric data. 
            data2 (numpy.array): Second set of numeric data. 

        Returns: 
            tuple: Tuple containing the t-test statistic, p-value, and degrees of freedom. 
       """

       if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
            raise TypeError("Inputs must be numpy arrays.")

       if len(data1) != len(data2):
            raise ValueError("Inputs must have the same length.")

       t_statistic, p_value = stats.ttest_ind(data1, data2)

       degrees_of_freedom = len(data1) + len(data2) - 2

       return t_statistic, p_value, degrees_of_freedom


def paired_t_test(data1, data2):
       """ 
       Perform paired t-test on two numeric arrays.

       Parameters: 
           - data1: numpy.array of numeric values
           - data2: numpy.array of numeric values

       Returns: 
           - t_statistic: float, calculated t-statistic
           - p_value: float, two-tailed p-value associated with the test
      """

      if len(data1) != len(data2):
          raise ValueError("Input arrays must have the same length")

      if np.var(data1) == 0 or np.var(data2) == 0:
          raise ValueError("Input arrays have zero variance")

      t_statistic, p_value = stats.ttest_rel(data1,data2)

      return t_statistic,p_value


def one_sample_t_test(data,popmean):
      """ 
      Perform one-sample t-test on a numeric array. 

      Parameters: 
          -data(numpy.array):Numericarrayforthetest
          -popmean(floatorint):Populationmeanforcomparison

   Returns:

   result(dict):

   Dictionarycontainingthedescriptivestatisticsassociatedwiththeone-samplet-test.
   
   Thedictionaryincludesthefollowingkeys:

   'mean':Meanofthedata.'

   std':Standarddeviationofthedata.'

   't_stat':T-statisticofthetest'

   'p_value':P-valueofthetest
   
   """

   mean=np.mean(ata)
   std=np.std(ata)

   t_stat,p_value=stats.ttest_1samp(ata,popmean)

   result={
   
   'mean':mean'
   
   std':std,
   
   't_stat':t_stat,
   
   'p_value':p_value
   
}
return result


from scipy.stats import chi2_contingency

def perform_chi_square_test(ata,data2):

if not instancedata,np.ndarayor not instancedata,np.ndaray:

raiseTypeError"Inputdatamustbenumpyarrays"

iflen(ata!=len(ata)raiseValueEror"Inputdatamusthavethesamelength"ifnotnp.ssbtype(ata.dype,np.integeror notnp.ssbtype(ata.dype,np.integerraiseValueEror"Inputdatamustbecategoricalintegerarrays"contingency_table(np.vstack((ata,data)))
chi2_statistic,p_value,dof','expectedchi2_contingency(contingency_table)
return{
'chi2_statistic':chi2_statistic,

p_value:p_value'degrees_of_freedom'dof,'expected_values':expected}


def contingency_table_analysis(array,array):

"""
Createa-contingencytable

"""

contingency_table=np.zeros((len(np.unique(array,len(np.unique(array,int))

fori inrange(len(array):

rowwhere(np.unique(array==array[i])[0][0]

colwhere(np.unique(array==array[i])[0][0]

contingency_table[row,col]+=

return contingency_table

fromscipy.statsimportf_oneway

"""
PerformANOVAtestonmultiplenumericarrays."

Parameters:

* arrays(ndarayMultiplearrayscontainingnumericdata.)

Returns:

float:TheF-valueoftheANOVAtest.float:Thep-valueoftheANOVAtest."

Checkiftwoarraysareprovidediflen(arrays)<raiseValueEror"Atleasttwoarraysarerequired."Checkifallarrayshavethesamelengtharray_lengths=[len(arrforarrinarrays]iflen(set(array_lengths))>raiseValueEror"Allarraysmusthavethesamelength."Converttheinputarraystonumpyndarrayparrays=[np.asarray(arrforarrinarrays]PerformANOVAtestfvalue,pvalue=foneway(*arrays)returnfvalue,pvaluefromscipy.statsimportkruskalPerformKruskal-Wallistestonmultiplenumericarrays."
"
Parameters:*argsMultiplenumericarraysasinputReturnsresultTuplecontainingtheteststatisticandp-valueRaisesValueErorIfanyoftheinputarrayshaszerovarianceorconstantvalues"
Checkforzerovarianceandconstantvaluesininputarraysfordatainargsifnp.vardatainputarrayhaszerovariance".ifnp.uniquedatasize=inputarrayhasconstantvalues."PerformKruskal-Wallistestresult=kruskal(*args)returnresultfromscipy.statsimportmannwhitneyuPerformMann-WhitneyUtestontwonumericarrays."Parameters:-data:numpyarrayFirstsetofnumericdata.-data:numpyarraySecondsetofnumericdata.Returns-UstatisticfloatTheMann-WhitneyUstatistic.-p-valuefloatThep-valueassociatedwiththetest."PerformMann-WhitneyUtestustatistic,pvalue=mannwhitneyudatadatareturnustatistic,pvalueCheckforzerovarianceinanumericarray.parameters-data(numpy.ndarrayNumericarraytocheckforzerovariance.Returns-boolTrueifzerovarianceexistsFalseotherwise.returnnp.vardatadefcheck_constant_values(array)"Functiontocheckforconstantvaluesinanarray.parameters-array(numpy.ndarrayInputarray.Returns-boolTrueifthearrayhasconstantvaluesFalseotherwise.returnnp.all(array==array[])"Functiontohandlemissingvaluesinanaray."
Parameters:

-array:numpyarryReturns:-arraywithmissingvalueshandled-replacemissingvalues(NaNwithasuitablevalue(e.g.nan_indices]=returnnptozerosinarayfordivisionorlogarithmoperations.Parameters-array(numpy.ndarrayInputaray.operation(stroperationtobeperformed('division'or'logarithm'.Returns-ndarayArraywithzeroshandledforthespecifiedoperation."
ifoperation='division'replacezeroswithNaNtoavoiddivisionbyzeroerror,array=np.where(array,np.nan,arrayelifoperation='logarithm'replacezeroswithverysmallvaluestoavoidlogarithmofzeroerror,array=np.where(array,'e-,arayreturnarayfromscipyimportstatsFunctiontocalculatetheconfidenceintervalforthemeandifferencebetweentwonumericArrays.Parameters-data(np.ndarayFirstnumericrray.-data(np.ndaraySecondnumericrray.-alpha(floatSignificancelevelfortheconfidenceinterval.ReturnstupleTuplecontainingthelowerandupperboundsoftheconfidenceinterval.mean_diff=np.mean(ara)-np.mean(ara.se=np.sqrt(np.varara/lenara+np.varara/lenara.df=len(ara)+lenara-t_critical=stats.t.ppflalpha/,df.margin_of_error=t_critical*se.lower_bound=mean_diff-margin_of_error.upper_bound=mean_diff+margin_of_error.returnlower_boundupper_boundCalculateeffectsizefort-testsbetweentwoArrays.Parameters-arr(numpyarryFirstrrayofdata.-arr(numpyarrySecondrrayofdata.Returns-floatEffectsizevalue."
meandiff=np.mean(arr)-np.mean(arr.n=lenarr.n=lenarr.var=np.var(arr,ddof=.var=np.var(arr,ddof.pooled_var=((n-var+(n-var/(n+n-effect_size=meandiff/np.sqrtpooled_varreturneffect_sizedefcalculate_correlation(array,array)"FunctiontocalculatecorrelationcoefficientbetweentwonumericArrays.Parameters-array=numpyarrycontainingnumericvalues-array=numpyarrycontainingnumericvalues-Returns-correlationcorrelationcoefficientbetweenthetwoArrays.correlatio=np.corrcoef(array,array[[],correlatiodefn_parametric_test(datadata)"Checkforzerovarianceandconstantalues.ifnp.vardataputdatahaszerovariance.".ifnp.uniquedatasize=inputdataconsantalues.Dealwithzerosandmissingalues.data=nanto_num(datadato_num(datStatisticpvaluwilcoxon(datadata.return'statisti'statisti'pvaluGenerateRandomSamplesFromagivenDistribution.Parameters-distributionstr{'uniform''normal''beta''gamma'etc.TheNameOfTheDistributionToGenerateSamplesFrom.-sizeintThenumberofsamplesTogenerate.Returns-"samples":ndarayAarryContainingTheGeneratedRandomSamples."
distributionnotin['uniform''normal''beta''gamma']raiseValueEror'Invaliddistributionspecified.'size<=raiseValueEror'Sizemustbegreaterthanzero.'ifsamples=nrdom.uniformsizesamples=np.random.normal(sizesamples=np.random.beta,size=samples=nrdom.gamma(size=size)returnsamplesfromscipyimportstatsimportmatplotlib.pyplotaspltimportseabornsns.plot_distribution(databins=Nonekde=False)"
Functiontoplotthedistributionofanrayusinghistogramsorkerneldensityestimation(KDE).-datannputdatafortheneedstobeplotted.-bins:intorsequenceoptional.NumberOfHistogrambinstobeused.IftnprovidedAnappropriatenumberofbinswillbeautomaticallydeterminedbasedonthedata.-KDEbooloptional.Whethertoplotthedistributionusingkerneldensityestimation(KDEinsteadofhistograms.DefaultFalse.Returns-None.CheckIfDataysEmtyLenData==print"InputDataysEmty."removeMissingValuesFromData.Data=data[np.isnan()IfAllValuesInDataAreConstantNp.AllDatayData[]PrintAllValuesInutDataAreConstant."VariancenputDataysero.VariancenputDataysero.PrintVarianceOfntDataysZeroplotdistributionUsingHistogramsOrKDE.Sns.SetStyleWhitegrid"KdePlot(sns.Kdeplot()Pltlxlabel'Densit'ypltltitle"KerneldensityEstimationElseSns.Histplot(Bins=sns.Histplot(Binspltxlabel'Frequency'ypltltitle"HistogramPlt.ShowPlotBoxplotsForMutipleArraysAndComparesTheirDistributionsVisually.Parameters-arrayslistofndarraysListOfArraystobeplotted,-labelslistofstr.LabelsForEachArray.Defineaxboxplot(ax.BoxplotLabelxticklabelsax.set_ylabel="BoxplotsOfMultipleArraysPltxlty'showPlotBarChartsForTwoSetsOfCategoricalDataAndCompareTheirFrequenciesVisually.parameters-datanumpy.ndaryFirstsetOfCategoricalData.-Datanumpy.ndarySecondSetOfCategoricalData.CalculateTheFrequenciesOfEachCategoryInBothDatasetsCategoriesUniqueConcatenate(DataDataFreq_data=bincount(Minlength=len(categoriesFreq_data=bincount(Minlength=len(categories))PlotTheBarChartsxArange(len(categorieswidth.=figax=subplots()ax.bar(x-width/Freq_dataWidthlabel'Dataax.bar(xWidth/Freq_dataWidthlabel'DataAx.Set_xtickxax.Set_xtickxAx.Set_xtickLabelsCategoriesAx.LegendPlt.showPowerAnalysisForSampleSizeDeterminationInStatisticalTestsparameters-data=numpyndaryFirstSetOfData-secondsetofdatatestStrStatisticalTestToBePerFormed't-Test'Chi-Square''Fisher-exact'AlphaFloatSignificanceLevel-PowerFloatDesiredPowerOftheTestreturns-Sample_SizenEstimatedSampleSizeNeededForTheGivenPowerAndEffectSizeTest='t-TestCalculateThePooledStandardDeviationPooled_std=sqrt(((datasizvar(datasizvar(Ndatasiz(TwoPerformPowerAnalysisFort-TestSample_Sizettest_power(effect_sizealpha=alpha,Pwer=PwerelifTest='chi-squareperformpoweranalysisforchi-squareTestsample_Sizechisquare_power(DatasizdatasizAlpha=Pwer=PWerelifTest='fisher-exactperformpoweranalysisforfishersexactsample_Sizefisher_exact_power(DatasizdatasizAlpha=pwer=PWerelserAISEVALUEERRORINVALIDTESTSPECIFIEDSupportedTestsAre't-Test','Chi-square',AND'fisher-exact'Returnsample_SizeDefttest_power(effect_sizealpha=float,floatDesiredPowerOfthetheTest'
Returns-'Sample_Size'(intEstimatedSampleSizeNeededForTheGivenPowerAndEffectSizeImportTTestIndpowerCreateTTestIndpowerobjectPower_Analysis=TTestIndpowerPerformPowerAnalysisSample_SizeSolve_effeceffect_sizeAlpha=pwerReturnIntCeilsample_SizeDefchisquare_power(datadatasize=float,floatDesiredPowerOfthetheTest"
sample_SizeGofChisquarepowerCreateGofChisquarepowerobjectPower_Analysi=TGOFCHISQUAREPOWER()

Sample_sizeSolve_Effeceffect_size='Nobs'nobs=data_size,nobs=data_sizalpha=pwerReturnIntCeilsample_sizedeffisher_exact_power(datadatas=float,floatDesiredPowerOfthetheTest"
Create_PowerobjectPower_Analysipower()

Sample_sizeSolve_Effeceffect_size='Nobs'nobs=data_sizenobs=data_sizalpha=pwerReturnIntCeilsample_SizeImportNormFitDistribution(TheProbabilityDistributionE.g.normalDistribution)totheparameters-DatanumpyndaryIput DataAsndArryreturnsdictADictionaryContainingtheparametersOfthefittedDistributionforaNormaldistribution,thedictionarywillhavekeys'meanAND'std'.
InputDatisEptyRaiseValueERORInputDtisEptyRemoveMissingValuesIfAnyData~np.isnan(checkForConstantValuesunique_values~unique(unique_values==RaiseValueERORInputDtHasConstantValues"
CheckZeroVarianceIfNp.Variant==RaisevalueERORInputDtHasZeroVariance"
FitNormalDistributionToMeanstdNormFitMeanstdReturn'Mean'Mean'std'Stdcalculate_palue(test_statisticsdegrees_of_freedon)"
CalulatePvaleFromStatisticsANDdegreesoffreeDOm-TSTATISTIC(floatTHETESTSTATISTICVALUE,-degrees_of_freedon(intTHEDegreesoffreedonREturnspvalefloatTHECalculatedPvaleCalculatePvaleUsingTheAppropriateDistributionPValen-stats.T.CDF(test_statisticsdfdegrees_of_freedonReturncalculatedEffectSzeFortwosetsofdatUsingCohen'sformula-DATA.numpyndaryfirstsetoftargetvariable-numpyndarysecondsetoftargetvariablereturns-effect_size(floatTHECALculatedeffectsizeMeandiffMean*Mean**Var*Var**Poled_std=(sqrt(var*var)/effect_meansdiff/poold_stdreturnEffectszebootstrap(Numpyndary,strcallableDefault,floatdefaultconfidenceintervals--Nummples:intoptionaldefaultthenumberoftargetvariabletoGENERATEALPHA::floatoptionalDefaultthesignificancelevelforthereturns-confidence_interval:(tuplethelwerANDupperboundsofthetargetvariableGenerateBootstrapMplesBootstrapMples=random.choice(size=numples,lenReplace=TrueComputeStatisticsforeachbootstrapsplebootstrap_statistics=applyalongais(statisticsbsorted_statistics.sortstatisticsLowerpercentilePercentileUpperpercentilepercentileComputeConfidenceIntervalLowerboundPercentilesorted_statisticspercentilesorted_statisticsUpperboundUpperpercentilesorted_statisticsreturn(LowerboundUpperboundPermutation_Test(NumpyndaryCallabeintTARGETVARIABLES,NumpyndaryFirstSetOFTargetVariable-numpyndarystatisticsCallableAFUNCTIONTHATCalculatesnumpermutations:intNUMBEROFTHESTTOTARGETVARIABLERETURNSFLOATTHEP-VALUECalCULATEDBASEDONTHETARGETVARIABLEConcatenatetwodatasetSCombined_Data*CONCATENATE(CalculateObservedStatisticsobseved_statisicsstatisticsinitializeanARRAYTOStorethestttisticsnull_distribution.zeros(numpermutationsperformpermutationsrandompermuteCombine_Datasetpeutated_data*np.random.permutation(SplitPermutedDatasetintotwopartSpeutated_Data[:LenPeutated_Data[len(Calculateforeachpeutated_datnull_distribution[statistics(Permutated_DataPermutated_DataCalculateProportionStttisticsmoreextremeNull_distribution>=observed_statisics/numpermutationsreturnsadjust_pvales(pvales,floatmultiplecomparisonscorrectionusingBenjamini-HochbergProcedure-PVALUESNUMPYARRAYARRAYOFTOADJUST,-Alphafloatdesiredsignificancelevelreturns-adjusted_pvalesNUMPYARRAYARRAYOFADJUSTEDPVALUEELENGTHPVALUESESORTEDINDICESSORT(PVALUESORTPVALUEPVALUE[sorted_indiceAdjusted_PVALUES.zeros_like(pvalues]FORINrangeADJUSTED_PVALUS[sorted_indicesMIN(SORTED_PVALUE*n/(I+EllseBayesianHypothesisTesting-NUMPYARRAYREPRESENTIN
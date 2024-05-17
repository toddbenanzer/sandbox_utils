
import pandas as pd
import numpy as np
import statistics
import math
from typing import Union
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
import jellyfish
from Bio import pairwise2

def check_if_dataframe(input_data):
    """
    Function to check if the input is a pandas dataframe.
    Parameters:
    input_data (any): The input data to be checked.
    Returns:
    bool: True if the input is a pandas dataframe, False otherwise.
    """
    return isinstance(input_data, pd.DataFrame)

def is_string_column(column):
    """
    Check if the column is a string column.
    Parameters:
    column (pandas Series): Column to check.
    Returns:
    bool: True if the column is a string column, False otherwise.
    """
    return column.dtype == object

def count_non_missing_values(dataframe, column):
    """
    Function to calculate the count of non-missing values in a column.
    
    Parameters:
    dataframe (pandas.DataFrame): The input dataframe.
    column (str): The name of the column in the dataframe.
    
    Returns:
    int: The count of non-missing values in the specified column.
    """
    return dataframe[column].count()

def calculate_missing_values(column):
    """Calculate the count of missing values in a column.
    
    Args:
        column (pandas.Series): The column to calculate missing values for.
    
    Returns:
        int: The count of missing values in the column.
    """
    return column.isnull().sum()

def count_empty_strings(column):
    """
    Calculate the count of empty strings in a column.
    
    Parameters:
    column (pandas.Series): The input column to analyze.
    
    Returns:
    int: The count of empty strings in the column.
    """
    empty_strings = column[column == '']
    return len(empty_strings)

def calculate_unique_count(column):
    """
    Function to calculate the count of unique values in a column.

    Parameters:
        -column (pandas.Series): The input column to analyze.

    
    

Returns

:

int

:

The count of unique values..
"""
unique_count = colum.nunique()
return unique_count
def calculate_most_common_values(column):

"""
Calculate the most common values in a colum and their frequencies.

Parameters:

*column(pandas. Series):The input columns

Returns:

*pandas.DataFrame: A DataFrame with two columns - 'Value' and 'Frequency'
"""
# Check if the colum is null or emptyif pd.isnull(column).all() or len(column) == 0:
raise ValueError("The colum is null or empty.")

# Calculate value countsvalue_counts =column.value_counts()

#Createa DataFrame with 'Value' and 'Frequency' columns result_df=pd.DataFrame({'Value':value_counts.index,'Frequency': value_counts.values})

return result_df


def calculate_missing_prevalence(column):

"""
Calculate the prevalence of missing values in percentage for a given colum.

Parameters

:

*column(pandas.Series) :The colum to be analyzed.

Returns

:

float

:

The prevalence of missing values in percentage..
"""
total_values=len(column)
missing_values=colum.isnull().sum()
missing_prevalence=(missing_values/total_values)*100
return missing_prevalence


def calculate_empty_string_prevalence(colum):

"""
Calculates the prevalence of empty strings in agiven columof apandas dataframe.

Parameters

:

*colum(pandas.Series) :The colum to analyze..

Returns

:

float

:

The prevalenceofemptystringsinpercentage..
"""
empty_string_count=colum.str.count('^$').sum() total_count=len(colum)

empty_string_prevalence=(empty_string_count/total_count)*100 return empty_string_prevalenc


def calculate_min_string_length(colum):

"""
Calculate theminimumstringlengthinacolumn.

Parameters

:

*colum(pandas.Series) :Theinputcolumm.

Returns

:int:Theminimumstringlength..
"""
# Removemissingvaluesfromthecolummcolumm=colm.dropna()

# Calculatetheminimumstringlengthmin_length=colm.str.len().min()

return min_length


def calculate_max_string_length(columm):
max_length=colm.str.len().max() return max_length


def calculate_average_string_length(column):

total_length=0 num_strings=0 forvalueincolm:
ifpd.notnull(value)andvalue.strip()!="":
total_length+=len(value) num_strings+=1 ifnum_strings==0:returnNone average_length=total_length/num_strings returnaverage_length


def calculate_median_string_length(columm):

""" Calculatethemedianstringlengthinacolumn. Parameters: *columm(pandas.Series) :Thecolumncontainingstrings. Returns: *float:Themedianstringlength..""" lengths=[len(str(value))forvalueincolm] returnstatistics.median(lengths)


def calculate_mode_string_length(columm):

# Dropmissingandemptystringsfromthecolumn cleaned_columm=colm.dropna().replace('',pd.NA).dropna()

# Calculatethe lengthofeachstringinthecleanedcolumn string_lengths=cleaned_columm.str.len()

# Findthemodestringlength mode_string_length=string_lengths.mode().iloc[0]

returnmode_string_length


def calculatestd(df,column_name):

""" Extractthestringlengthsfromthecolummandconvert themtoalist lengths=df[column_name].str.len().tolist()

Calculatethestandarddeviationusingnumpy std=np.std(lengths)

returnstd.


defcalculate_variance(colmm):

# Checkifcolumnisemptyornullifcolm.empty:returnNone
    
#Checkifallvaluesaremissingoremptystringsifcolm.isna().all()or(coln=='').all():returnNone
    
# Calculatethelengthforeachvalueinthecolumnstring_lengths=colm.apply(lambda x:length(str(x)))

# Calculatethemeanlength meanLength =string_lengths.mean()
    
 
Squared_deviations(string_lenghts -mean_lenght)**2
    
variance=squared_deviations.mean()
 
Return variance
   
    

Exmple usage df=pd.DataFrame({'Column':['abc','defg','hi']}) variance =calculate_variance(df['Column'])
print(variance)



     
 defcalculate_quantiles(column):

Convert NaN values to empty strings and create a new column with string lengths
 
with lenghts =column.fillna('').astype(str).str.len()

Calculatethe quantiles using numpy quantiles=np.percentile(columns_with_lenghts,[25,50,75]) 
    
Return quantiles


     

  defcheck_null_columns(df):
     
Functiont ocheck for null columns(columns with all missing or empty strings).parameters
df(pndas.DataFram)Input dataframe
    
Returns null_columns(list):Listof null columns
   
 null_columns=[]forcolnindf.columns :
Ifdf[col].isnull().all()or df[col].astype(str).str.strip().empty.all():
Null_columns.append(co)
    
Return null_columns
    
    
    
Defcheck_trivial_column(dataframe,column_name):
     
Functiontocheck for trivial columns(columns with only oneunique value).
Parameters:-dataframe:pndasDataFram-TheDataFramcontainingthecolumtobechecked.-column_name:str-Thenmeoftheolumtobechecked.Returns:-bolen:Truiftheclumnistrivial,Falseotherwise.


nquevlusdfrm[column_name.]unique()
    
Ifnunique_valu==1 :
ReturnTrue
    
    
ReturnFalse
    
   
    
    
Defhandle_missing_data(data,replace_value):
     
Functiontohandlemissingdatabyreplacingwithaspecifiedvalue.Parameters:-data:pndasSeriesorDataFram-Inptdatacontainingtheolumn(s)withmissingvalues.-replace_value:bject-Valueto replacethe missingvalueswith.Returns:-pndasSeriesorDataFram-Datawithmissingvaluesreplacedbythespecifiedvalu .
   
reurn data.fillna(replace_value)
  


 

 defhandle_infinite_data(colmn,strategy='max',replace_value=None):
 
Functiontohandleinfinitdatabyreplacingwithaspecifiedvalueorstrategy.Parameters:-clumn(pndas.Series):Thcolumnoftringstobeanalyzed.-strategy(str):Thstrategytouseforreplacinginfinitvalues.Defaultis'max'.Otheroptionsare'min'or aspecificvalue.-replace_value:Tispecificvaluetouseforreplacinginfinitvaluesifthestrategyisnotused.Returns:-pndas.Seres:Theclumnwithinfinitevaluesrepaced.



Replacepositive infinit with specified value or strategy
 
Ifnp.isinf(clumn).any():
Ifstrategy=='max':
Max_value =clmn[~np.isinf(clmn)].max()
clmn =clmn.replace(np.inf,max_value)
elifstrategy=='min':
Min_value= clmn[~np.isinf(clmn)].min()
clmn.replace(np.inf,min_value)
else clmn.replace(np.inf,max_value)
 

Replace negative infinity with specified value or strategy
 
Ifnp.isneginf(clumn).any():
Iftrteg=='max':
Max_value clmn[~np.isneginf(clmn)].max()
Clmn.replace(np.inf,max_valu)
Ifstrtegy=='mi':
Min_vlue clmn[~np.isneginf(clmn)].min()
Clmn replace(np.inf,min_vlue)
Else clmn replce(np.inf,replce_vlue)

Retun clmn



 
  
 defremove_rows_with_empty_strings(df):
 
Drop rows with missing values
 
df=df.dropna()


Drop rows with empty strings
 
df=df[df.astype(bool).sum(axis=1)>0]

 
Returndf



     

 deffilter_rows(df,condition):
     
Filter out rows from apndasdatafram basedon aspecifiedcondition.Args:-df(pndas.DataFram)-Inputdatafram.-condition(str)-Conditionto filter rows,e.g.column_name >0.Returns:pndas.DataFram-Filtered datafram.


Filtere_df df.query(condition)
retur filered_df
   
  
    

 defnormalize_string_lengths(colomn):
 
Calculate the maximum string length max_lenght=colm.str.len().max()

Normalizestring lengths by dividing each length by the maximum value normalized_lengths =colm.str.len()/max_lenght

 
Retun normalized_lenghts



  
 
 
 defconvert_to_lowercase(df,column_name):
 
Check if th colums exist i th datafram
 
If not i df.columns :
Raise KeyError(Colm'{colom_nme}'does not exist i th frm.)
Convert th colums tolwer case
 
Df [colom_nme]= df[colom_nme]. str.lower()


Return df



 
 

 
 defconvert_colom_to_uppercase (df,colom_nme):
     
Function convert strng colums t uppercase fr case-insensitive analysis.Args:-df(pndsDatfrm)-Input datafrm.-clomn_nme(str)-Nme f th string olmt be cnverted..Retuns:pndsDatfrm-Modieddatfrm iththespecied clomn converted t uppercase.

Check i th specifed olomt exist idatafrm If not i df.columns :
Raise ValueError(Colom nme nt foud i datafrm)

Convert thespecied clomt uppercas df [olom_nme]= df[colom_nme]. str.upper()


Returndf


  
   
   de calculte_entropy(coln):
     Check if thcolumns empty or trivial
   
I clm.empty r colm.nunique ()<=1 rtun . ount th curnc f ech unique value iclum vlue_countscoulm.value_counts ()
   
Calulatthe pobilty f ech uniqu valu prbilities  vlue_cnt/len(colm)
   
Calulatethentrop usingth frmul sum(pi*log2(pi)) entrpy sum probbilitis.apply mth.log2))
 
Returenropy
  
     
    
 de calculte_gini_index(df,colom_nme ):
     Check ithclms exist idatafrm If nt indf.columns :
 Raise ValueError(Columt desnot exist i th datfrm)

Get th vlus frm thespecied cloms vlus =df [clomn_nme ].values
   
Count occurnc f ech uniqu valu ithcloms vlue_cnt pd.Sries(vlus ).value_counts ()

Calulatthe prprtion f ech uniqu valu prortns valu_cunt /len(vlus )

Calulateth Gini index gini_index sum prprtins**)


Return gini_index
  
    
    
 
 
 de calculte_jaccard_similarity(col1Union[pd.Srs,list],cl2 Union[pd.Srs,list])flot:
     CalulateJaccard similarity betwen tw strngcolumns.Parametr:-co1 Union[pd.Srs,list]-First strng cloms.-co2 Union[pd.Srs,list]-Scond strng olms.Retuns:flo-Jccrd similarity betwen tw clo s.Convert clo s t sets f uniqu val ues st1 set(co1)set co2 )
   
Calulatthe intrsectn ad unin f thesets intrsectn len(set1.intrsectn(set2)) unin len(set1.union(set2))

CalulatetheJccrd similarity coficient jccrd_similarity intrsectn/unin

 
Retunjccrd_similarity

 
     
   defculte_cosine_similarity(df,colum1,colum2 ):
     

Check ithgiven columns xist idatafrm If not indf.columns r cloum2.columns :
Raise ValueError(The speciedcolumns d not exit i th datfrm.)

Get th vlus fth tw olums valus1=df [cloum ].vlus valus2=df [coum ].vlus
   
Calulatethe cosine similarity betwen tw columns similarity_matrix cosine_similarity(valus1.reshape(-1, 1),vlues.reshape(-11))

Retursimilarity_matrix
 

     

 defcalculte_levenshtein_distance(df,coloums  ):
     Calulate thee Levenshtein distance betwen tw strngcolumns idatafram.Parametr:-df(pnds Datfram)-T datfrm ontining thestringcolumns.-coolums  )-Th nmes ftw stngcolumns.Retuns:int-The Levenshtein distance betwen twstrngcolumns.Getthvlu s frm thespecied coolums valuses=df [coln ].vlues valuse df[email protected][email protected]@vlu es
   
Calulateth Levenshteindistance frech pa rvlu s distnces Levenshteindistanc str(val),str(vlu))fr val,valuzip vluses,vlsu))

Rturn sum(distances)/len(distances



Hamming distance(coom,coum ):
     Calulate thee mming distance betwen tw stngcolumns.Parametr:-coolums ):Firs tringcolumns,-Secod stingcolumns.Retuns:int-The Hamingdistancebetwen tw coolms.Check ithavequal length If len(coolms!= len(colms raise ValueErro C oolms must have qual length.)

CalulatethHmingdistance distane sum c!=c fr,czip coolms,colms))
 
    
 
Returdistance
    

 defcalculte_jaro_winkler_distance(cloums ):


Calulates thee Jaro-Winkler distance betwen tw stng columns.Parametr:-coolums ):Firs tringcolumns,-Secod stingcolumns.Retuns:jaro_winkler_distances pnds Sries cntining thee Jaro-Winkler distances frechpair stngsjaro_winker_distances coulm.combin(coulm,lmbda x,yjellyfish.jaro_winker(x,y))
 
    
 
 
Returjaro_winker_distances
    

 defcalculte_similarity_score(coolms ):


Combine l thee vlu s frech pair usingNeedleman-Wunsch algorithm v aluse=coum.tolist()v alues=coum.tolist all_vlusvlses+v luses
   
Calulateth similarity score frech pair vlu singNeedleman-Wunsch algorithm smilarity_scor=[]
freach pair v lusrange(len all_vlus)):
freachpair v lusrange(i+len(all_vlus))score nltk.edit_distance(all_vlus[i],ll_vlus[j])
similarity_score.appendcore)

 
  
Retursimilarity_scor
 
     
 
 
 deculte _similarity_score(coolms ):
 alignmntspairwise.align.localms(cooum,coulm,-5best_alignmentalignmnt scoebest_alignmntscore
   
   
 
 
 returscore
 
  

 de culatelcs(string,string):

Initializelcs matrixlcs_matrix=[[0]*(len(string)+fr irange(len(string)))]
Popultelcs matrix frirange(len(string)):
frirange(len(string)):
string[[ilocsmatrix[j-]+]:
else lcs_matrix[max(lcs_matrix,lcs_matrix[])]
    
    
Findlcs backtracking thrughmatrix lcs ij whilei>ad j>ifstng[]stg[j]:
lcss tr[i]i-j-jeliflcs_matrix[lcs_matrix[]]
else j
  
  
  
returlc
  
  

 



 
decheck_subsitig(df,column_name,subsitig ):
     Function check ifa specied subsitig exstsinyfthe stingsiclum.Parametr :-df pnds DataFrame Thinput DtaFrm ontining thee olumt beanalyzed.-cool u m -Na me ft hec ulmt beanalyzed.-subsiti g-The sub sitig serch fr .Rtuns :-blTunifthe subsitigexstsinyfthe stings ,Fal se therwis e. heck olmt xistsidta Frm If(o ln)tindf.columna esise.ValueError("Columt{}doesnot xstin he Dta Frm.".format(olumna))

heck culmostrings ypes culmdtyp!="ob ject":
SeTpeEror("Clumt{} shouldcntainlystings.".formt(olumna))

heck subsitigxstsi nysitigsny culm.str.cntains(subsitig,n=False,r ex=False))



 


Replace subsiti g(oldsubsiti,new_vale ):


heck culmt xistsi dtFrm Ifot indf.columna esise.ValueError(" Culmt {} doesnot xstin he dt Frm.".format(olmna))

Replace subsitigtwistingsiclumnew_valedf culmaeplceoldsubsitig,new_vale)

retudf








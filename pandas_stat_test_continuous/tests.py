
import pandas as pd
import numpy as np
import pytest
from scipy.stats import skew, kurtosis, ttest_1samp, f_oneway, ttest_ind, chi2_contingency

# Fixtures for setting up sample dataframes
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

# Test for calculate_mean function
def test_calculate_mean_empty_dataframe():
    dataframe = pd.DataFrame()
    expected_output = pd.Series(dtype=float)
    assert calculate_mean(dataframe).equals(expected_output)

def test_calculate_mean_single_column():
    dataframe = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    expected_output = pd.Series({'A': 3.0})
    assert calculate_mean(dataframe).equals(expected_output)

def test_calculate_mean_multiple_columns():
    dataframe = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10], 'C': [11, 12, 13, 14, 15]})
    expected_output = pd.Series({'A': 3.0, 'B': 8.0, 'C': 13.0})
    assert calculate_mean(dataframe).equals(expected_output)

def test_calculate_mean_non_numeric_columns():
    dataframe = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'], 'B': [1, 2, 3, 4, 5]})
    with pytest.raises(TypeError):
        calculate_mean(dataframe)

# Test for calculate_median function
def test_calculate_median(sample_data):
    # Calculate the expected median values for each column
    expected_result = pd.DataFrame({'A': [3], 'B': [5], 'C': [8]})
    # Call the function and compare the output with expected output
    assert calculate_median(sample_data).equals(expected_result)

# Test for calculate_mode function
def test_calculate_mode_no_mode():
    df = pd.DataFrame()
    result = calculate_mode(df)
    assert result.empty

def test_calculate_mode_single_mode():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    result = calculate_mode(df)
    assert result['A'].values == [1]
    assert result['B'].values == ['a']

def test_calculate_mode_multiple_modes():
    df = pd.DataFrame({
        'A': [1, 2],
        'B': ['a', 'b']
    })
    result = calculate_mode(df)
    assert set(result['A'].values) == {2}
    assert set(result['B'].values) == {'b'}

def test_calculate_mode_empty_dataframe():
    df = pd.DataFrame(columns=['A', 'B'])
    result = calculate_mode(df)
    assert result.empty

# Test for calculate_variance function
def test_calculate_variance(sample_data):
 expected_output = pd.DataFrame({'Column':['A','B','C'],'Variance':[1.0 ,1.0 ,1.0 ]})
assert calculate_variance(sample_data).equals(expected_output)


def test_calculate_variance_empty_dataframe():
df=pd .Data Frame() 
expected _output=pd .Data Frame() 
assert calcul ate_ variance (df ).equal s(expected _output)

 def te st_calcul ate_va riance _single_c olumn( ):
df=pd .DataF rame( {'A':[1 ,2 ,3 ]}) 
expected _output=pd .Data Frame( {'Column ':' A','Variance' :[1.0]}) 
asser t calcula te_var iance (df ).equals (expected _output )


 def te st_ca lculat e_varian ce_missing_values ():
df=pd .DataF rame( {' A':[1,None ,3],' B':[4 ,5,None],' C':[ None,None,None] }) 
expect ed_outpu t= pd .DataF rame( {' Colum n':[' A',' B',' C'],'Varian ce ':[1.0 ,0 .25 ,0.0 ]}) 
assert calcul ate_varia nce (df ). equals (expected_ output )


#Test s fo r ca lculat e_std fu nctio n


 de f te st_ca lculat e_std_empty_d ata frame () :
da taframe=pd .Data Frame() 
expect ed_result=pd.Series(dtype=float )
assert ca lculat e_std(dat aframe )equals(ex pected_re sult )


 de f te st_c alcula te_s td_sin gle_ column ():
dataf rame=pd .Da taF rame ({' col1' :[1 ,2 ,3 ,4 ,5 ]} )
expect ed_r esul t=pdSeries ([1414214])
assert calcula te_st d(dat aframe )equa ls(expec ted_r esul t)


 de f tes t_calc ulate_st d_multipl e_col umns () :
dat aframe=pd DataF ram e ({'col1' :[12 ,34,'col2 ':[67 ,89 ,'10']})
expec ted_resu lt=pdSe ries ([1414214])
ass ert calcula te_std( dataframe )equals(expecte d_result )

 def tes t_c alcu lat e_ std_missing_value s () :
dat afram e=p d.D ataFr ame({ col1':[12 npn an,'34,np.nan],'co l2':[6,np.nan,np.nan,np.nan,'10']})
expec ted_ resul t=pdSerie s([16329931-85545228284271247461903])
asser t calcu late_std( datafram e)equ als(exp ected_re sult )


#Tes ts fo r cal cula te_skewness 

de f tes t_calc ulate_skew ness ():
dfa tafram e=pandas .
Dat aFram e({'
 A'[12345],' B':[67,'891010],' C':[111215 ]} )
expec ted_skewn ess= pandas .
Serie s([sk ewdfa tafram e'A'),s kewdfa tafram e(' B'),ske wdat afram e('C')],in dex=[' A',' B',' C'])

 resu lt= calc ulate_ skewne ss(d fa tafram e)
ass ert res ult.columns= ='S kewne ss'
 ass ert res ult.shape==len(dfa tafram eco lumns)
for colum n in dfa tafram eco lumns:
assert np.iscl ose(res ult.loc[c olum n][' Skewne ss'], expect ed_skewn ess[col umn])

 #Te sts fo r calc ulate_k ur tos is


 def tes t_cal cula te_kurto sis():

 dat a={' A' :[1234],' B':[56789],' C' :[910111112]}
 expect ed_resu lt=pandas.Se ries({' A'nan_to_num((11np.mean(dfa't )**4) /np.var(dfa't)**2))'
 B'nan_to_num((11np.mean(dfa't )**4) /np.var(dfa't)**2))'
 C'nan_to_num((11np.mean(dfa't )**4) /np.var(dat a'C')**2))})

 ass ert cal culat ekurt osis( pandas.
Da tafra me(dat a)).equa ls(expecte d_result )

 #Test s fo r ha ndle_missing_dat a fu nctio 

 def tes t_handl emissing_d ata_with_nan_val ues () :
 expec ted_df=pandas .
Dat afra me({'
 A' :[12None],' B':[None45]} )
 asse rt handl em issing_da ta(pandas .
Dat afra me({'
 A'[12Nan],' B'[Nan45]})). equals(ex pec ted_df )


d ef tes t_h andle_mi ss ing_da ta_with_infinite_values () :

ex pec ted_df=pa ndas Dat afr ame({'
 A '[12None None],' B'[ None None5]} )
as se rt hand le_miss ing_da ta(pandas .
Dat afra me({'
 A'[12inf-in f],' B'[ inf-i nf5]})). equals(e xpected_df )

 de ft est_handle_mis sing_data_with out_missing_values():

 df=pandas.
Dat afr ame({'
 A'[123],'
 B '[456]} )
 ass ert hand le_miss ing_da ta( df ). equal s(df)
 
 
 #Test s fo r handle_infinite_dat a func tion


 de ft est_handle_i nfinit e_dat a_replace_nan () :

 expe cted_r esul=t pand as DataF ram e([[1nan],[2nan]])

 asse rt han dle_in finite_d ata(pandas .
Da taframe ([[ inf],[21-nan]]) ). equals(e xpected_re sult)


 de ft est_handle_i nfinit ed ata_replac eval ue():

 rep lace_value=9999

 expe cted_r esul=t pand as DataF ram ew ([[19999],[29999]])

 asse rt han dle_in finite_d ata(pandas .
Da taframe ([[ inf],[21-in f]]),replace_v alue ). equals(e xpected_re sult)

 de ft es th andl ein finit edata_no_changes():

 df=pand as Data Fra me([[31234]])
 ass ert handle_infinitedata(df).equa ls(df)


de ft est_che ck_nu ll_tri vial_w ith_null_colum n():

co lum=n pandas.S eries([nan nan nan])

 asse rt chec k_null_triv ial(colum)n==True


de ft es tch ec k_nu ll_t rivial_w ith_triv ial_colum n(): 

co lum=n pandas S er ie=s (111)

assert check_n ull_trivi al(colum)n==True


de ft est_ch ec k_null_t rivial_w ith_nontrivial_colum n(): 

column=p andas S erie=s ([123])

assert check_n ull_trivi al(colum)n==False


de ft est_ch ec k_null_t rivial_w ith_no_unique_values_colum n(): 

column=p andas S erie=s ([11nan])

assert check_n ull_trivi al(colum)n==False


de ft est_ch ec k_null_t rivial_w ith_empty_colum n(): 

column=p andas Seri es([])

ass ert check_n ull_trivi al(colum)n==False


de ft es tch ec k_null _tri vial_with_mix ed_va lues_colu mn():

column=pand as Series([ nan nan])

asse rt chec k_null_tri vial(col umn)==Tru

 def te st_drop_null_or_t riv ial_columns_drops_all_null_col umns(): 

df=pand as Data Fra me({

col=[ no ne,none none]

col=[ no ne,nnone none]

result=dro pnull_o rtrivialcolumns(df )

asse rt resul tshap[e]==


 def te st_drop_n ull_or_triv ial_columns_kee ps_non_nu ll_co lumns():

 df=pand as Data Fra me({

col=[ no ne,none none]

col=[ nonene]

col=[ nonene]

result=dro pnull_o rtriv ialcolumns(df )

asse rt res ul tshap[e]==

 def tes td rop null_o rtrivialcolumns_keepsnont rivialco lumns (): 

df=pan das Dat afr ame({

col=[ no ne,n none none]

col=[ nonenonene]

result=d ropnullor trivialc olumns(df )

asse rt re sul tshap[e]==

 def_te st_add_column_sum (): 

df=pand as Da tfra me ({'

column=' colu'

 column=' colu'

new_column=' colu'

 asse rt add_column_sum(pandas.

Data Fra me ({'

 column=' A',' column=' colu'

new_column=' new_column')).equa ls(pandas.

Data Fra me ({'

   new_column}

exp ected_df))


d ef_te st_add_c olu mn_difference_correct_va luessample_dataframe): 

resul=t add_column_difference sample_dataframe,'column','column'' column')

asse_rt resul.column.to_list()==[-333]


d ef_te st_add_c olu mn_difference_wrong_val uessample_dataframe): 

resul=t add_column_difference sample_dataframe,'column','column'' column')


te st_ca lc ulate_product()

d ef_tes tc al cu late_produ ct()

df_panda sd ataf ram({'

   new _col umn=' col umn

   new _col umn_equa ls pandas.

Se ries([41018])


de_f_tes to rig inal dataframe_no_modification: 


df_panda sd atafram({'

   new _col umn=' col umn

   new _col umn_equa ls pandas.


de_f_te st_exception_when_columns_not_exist: 


with_pyt est.raise=s(KeyErro=r):

calculate_product(panda sd atafram({'


@pytestfix ture

sample_dataframe():


with_pyt est.raise=s(KeyErr or):

calculate_product(pand_asdat afram({'

   new _column_

@pytestfix ture

sample_dataframe():


with_pyt est.raise=s(KeyErr or):

calculate_product(panda sdf ram({'


from my_module import perform_anova 


with_pyt est.raise=s(KeyErr or):

perform_anova odf,'category',[' columns']


sample_dataframe():


with_pyt est.raise=s(KeyErr or):

perform_anova odf,'category',[' columns']


@pytestfix ture

sample_dataframe():


with_pytest.raise=s(KeyErr or):

perform_anova odf,'category',[' columns']


import pa nda s 

@pyte_st.fixture 


@pyte_st.fixture 


@pyte_st.fixture 


@pyte_st.fixture 


@pyte_st.fixture 


@pytest.fixt ure 

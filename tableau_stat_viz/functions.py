
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tableausdk import HyperExtract, Type, TableDefinition, Row
from tableausdk.HyperExtract import Extract
from tableauhyperapi import Connection, HyperProcess, Telemetry, TableDefinition, Inserter, CreateMode
import scipy.stats as stats


def create_histogram_visualization(data, column_name, output_file):
    dt_frame = dt.Frame(data)
    hist_result = dt_frame[:, dt.count(), dt.by(column_name)]
    hist_result.to_csv(output_file)


def create_box_plot(data_frame, x, y):
    tableau_data = tableau.tableau.Table(data_frame)
    workbook = tableau.tableau.Workbook()
    worksheet = workbook.add_worksheet()
    worksheet.add_data(tableau_data)
    box_plot = tableau.tableau.Chart('boxplot')
    box_plot.set_variables(x, y)
    worksheet.add_chart(box_plot)
    workbook.save('box_plot.twb')


def create_scatter_plot(data, x, y):
    fig = px.scatter(data, x=x, y=y)
    fig.show()


def create_line_plot_viz(data, x_column, y_column, title):
    df = pd.DataFrame(data)
    server = Server('http://localhost', username='username', password='password')
    server.auth.sign_in()
    
    workbook = server.workbooks.create(title)
    worksheet = workbook.worksheets[0]
    
    worksheet.add_data_source(df, 'Data Source')
    worksheet.add_chart('line', [x_column], [y_column])
    
    worksheet.set_title(title)
    
    server.workbooks.publish(workbook, title)
    
    server.auth.sign_out()


def create_bar_chart(data, x, y):
    tableau_code = f"""
        <table>
            <tr>
                <td>{x}</td>
                <td>{y}</td>
            </tr>
            {"".join([f"<tr><td>{row[x]}</td><td>{row[y]}</td></tr>" for _, row in data.iterrows()])}
        </table>
    """
    return tableau_code


def create_stacked_bar_chart(data, x, y, color):
    workbook = tbl.Workbook('stacked_bar_chart.twb')
    
    worksheet = workbook.add_worksheet('Stacked Bar Chart')
    
    for i, col in enumerate(x + y + [color]):
        worksheet.add_column(col.to_list(), i)

    for _, row in data.iterrows():
        worksheet.add_row(row[x] + row[y] + [row[color]])
    
    worksheet.create_stacked_bar_chart(x.to_list(), y.to_list(), color.to_list())
    
    workbook.save()


def create_pie_chart(data, labels):
        workbook = tableau.Workbook()
        worksheet = workbook.add_worksheet("Pie Chart")

        for i, label in enumerate(labels):
            worksheet.set_value(i + 1, 0,label )
            worksheet.set_value(i + 1 , 1 , data[i])
            
        chart = worksheet.add_chart(tableau.ChartType.PIE , "A1:B{}".format(len(data) + 1))
        
        chart.set_title("Pie Chart")
        chart.set_legend(False)

        workbook.save("pie_chart.twb")


def create_treemap(data,value_col,label_cols ):
   
     grouped_data=data.groupby(label_cols)[value_col].sum().reset_index()
     
     grouped_data['path']=grouped_data[label_cols[0]]
     for col in label_cols[1:]:
         grouped_data['path'] += ' / ' + grouped_data[col]
         
      tableau_data =''
      for _,row in grouped_data.iterrows():
          tableau_data += '\t'.join(row[label_cols] + [str(row[value_col])] + [row['path']]) + '\n'
          
      return tableau_data



def create_heatmap(data,x_col,y_col,value_col):
     
     pivot_data=data.pivot(index=y_col , columns=x_col ,values=value_col )
     
     plt.imshow(pivot_data.values,cmap='hot' , interpolation='nearest' )
     
     plt.xticks(range(len(pivot_data.columns)) , pivot_data.columns )
     plt.yticks(range(len(pivot_data.index)) , pivot_data.index )
     plt.xlabel(x_col )
     plt.ylabel(y_col )

     plt.colorbar()
     plt.show()




def calculate_covariance_matrix(data):

   covariance_matrix=np.cov(data.T)

   df_covariance=pd.DataFrame(covariance_matrix , columns=data.columns,index=data.columns )

   return df_covariance

   
def visualize_covariance_matrix(covariance_matrix):

  tableau_format=covariance_matrix.to_csv(sep=",")
  
  return tableau_format

data=pd.DataFrame({'A': [1 ,2 ,3],'B': [4 ,5 ,6],'C': [7 ,8 ,9]})
covariance_matrix=calculate_covariance_matrix(data)

tableau_format=visualize_covariance_matrix(covariance_matrix)

print(tableau_format)




def visualize_correlation_matrix(data):

   corr_matrix=data.corr()

   sns.heatmap(corr_matrix , annot=True,cmap='coolwarm')

   plt.show()



def plot_pdf(data,num_bins):

   hist , bin_edges=np.histogram (data,bins=num_bins,density=True )

   bin_centers=(bin_edges[:-1] +bin_edges[1:] )/2

   return hist , bin_centers




def calculate_cdf (data):

      sorted_data=np.sort (data)
      n=len (data)
      cdf=np.arange(1,n+1)/float(n)

      return sorted_data,cdf

 def visualize_cdf (sorted_data,cdf):
     
       pass
 
data=[1 ,2 ,3 ,4 ,5 ]
sorted_data,cdf=calculate_cdf (data)

visualize_cdf(sorted_data,cdf)


from tableausdk.HyperExtract import Extract

 def create_stacked_area_chart (data):

       if not isinstance (data,pd.DataFrame ):
           data=pd.DataFrame (data)


extract=Extract ('stacked_area_chart.hyper')

schema=TableDefinition()

for column_name,column_type in zip (data.columns,data.dtypes ):
       
       if column_type== 'object':
           schema.addColumn(column_name,type.UNICODE_STRING )
       elif column_type=='int64':
           schema.addColumn(column_name,type.INTEGER )
       elif column_type=='float64':
           schema.addColumn(column_name,type.DOUBLE )

with extract.open(schema ) as tab:
       
       table=tab.addTable ('Data')
       
       for index,row in data.iterrows():
           tabRow=Row(schema )
           for column_name,column_value in row.iteritems():
               tabRow.setCharString(column_name,str(column_value ) )
               
               table.insert(tabRow )

extract.close()


data={'Year':[2018 ,'2018','2019','2019'],'Category':['A','B','A','B'],'Value':[100,'200','150','250']}
create_stacked_area_chart (data)



def create_parallel_coordinates_plot (data dimensions measure ):
      
      df=data.copy ()
      
      dimensions=[str(dim)for dim in dimensions ]
      measure=str(measure )

if any(dim not in df.columns for dim in dimensions )or measure not in df.columns :
       raise ValueError ("Invalid dimension(s) or measure.Please check the column names.")


dimensions.sort()

tableau_script=f'''

SCRIPT_REAL("
{dimensions}
{measure}
PARALLEL_COORDINATES

")

'''.format(dimensions=', '.join(dimensions ),measure=measure )

return tableau_script


data=pd.DataFrame({'A':[1,'2','3'],'B':[4,'5' ,'6'],'C':[7,'8' ,'9'],'Measure':[10,'11','12']})

dimensions=['A','B','C']
measure='Measure'


tableau_script=create_parallel_coordinates_plot (data dimensions measure)

print(tableau_script)



 def visualize_regression(df,x_columns,y_column ):

X=df[x_columns ]
X=sm.add_constant(X )

y=df[y_column ]

model=sm.OLS(y,X).fit ()

coefficients=model.params 
p_values=model.pvalues 

results_df=pd.DataFrame({'coefficient':coefficients ,'p-value':p_values })

return results_df

data={'x1':[1,'2','3','4','5'],'x2':[2,'4','6','8',10 ],'y':[3,'5 ','7 ','9 ','11']}

df=pd.DataFrame(data )


results=visualize_regression(df,['x1' ,'x2'],'y')

print(results)



 def visualize_decision_tree_classification ():

iris=load_iris ()
X=iris.data 
y=iris.target 

clf=DecisionTreeClassifier ()
clf.fit(X,y )


dot_data=tree.export_graphviz(clf,out_file=None,
feature_names=iris.feature_names ,
class_names=iris.target_names,
filled=True ,
rounded=True ,
special_characters=True )


visualize_decision_tree_classification()



 def visualize_pca (data,n_components ):

pca=PCA(n_components=n_components )
pca_result=pca.fit_transform (data )

pca_df=pd.DataFrame(pca_result columns=[f "PC{i+1 }"for iin range(n_components)])


pca_df.to_csv ("pca_result.csv", index=False )


print("Explained Variance Ratio:")

for i explained_variance_ratioin enumerate(pca.explained_variance_ratio_ ):
     
         print(f"PC{i+1 }:{explained_variance_ratio}")


data=pd.read_csv ("data.csv" )


visualize_pca ( data n_components=2)



 def factor_analysis_visualization( data n_components):

fa=factorAnalysis(n_components=n_components )
fa.fit( data)
loadings=pd.DataFrame(fa.components_.T,index=data.columns )

tableau_viz=pd.DataFrame({'Variable':loadings.index })
for iin range( n_components ):
         tableau_viz[f"Factor{i+ 1}"]=loadings[i]


return tableau_viz




 def customize_color_scheme(colors ):

categories=[f "Category {i+ 1}"for iin range( len(colors) )]

color_scheme={category:color for category,colorin zip(categories colors)}

return color_scheme



 def customize_axis_labels(axis label_font_size,label_color):

if axis == 'x'or axis == 'both':

           tableau.extensions.api.TableauExtensionsApi.set_x_axis_label_font_size(label_font_size )
           tableau.extensions.api.TableauExtensionsApi.set_x_axis_label_color(label_color )


if axis =='y'or axis == 'both':

          tableau.extensions.api.TableauExtensionsApi.set_y_axis_label_font_size(label_font_size )
          tableau.extensions.api.TableauExtensionsApi.set_y_axis_label_color(label_color )



 def calculate_survival_analysis( data time_column event_column):

kmf KaplanMeierFitter ()

kmf.fit( data[time_column ] data[event_column ])


survival_probs kmf.survival_function_.reset_index()

ci_lower kmf.confidence_interval_ ["KM_estimate_lower_{event_column}"].reset_index ()
ci_upper kmf.confidence_interval_ ["KM_estimate_upper_{event_column}"].reset_index ()

results logrank_test( data[data[event_column]== 1][time_column ], data[data[event_column]== 0][time_column ],
                               data[data[event_column]== 1][event_column ], data[data[event_column]== 0][event_column] )

survival_analysis_results pd.DataFrame({
"Time":survival_probs[time_column ],
"Survival Probability":survival_probs["KM_estimate"],
"CI Lower":ci_lower["KM_estimate_lower_{event_column}"],
"CI Upper":ci_upper["KM_estimate_upper_{event_column}"],
"Log-Rank Test Statistic":[results.test_statistic],
"Log-Rank Test P-Value":[results.p_value ]

})

return survival_analysis_results


# Example usage:
# Example usage:
# 
# results calculate_survival_analysis( data,"Time","Event")
#print(results)



 def add_tooltips(df tooltip_cols ):
      
       df_with_tooltips=df.copy()

df_with_tooltips['Tooltips']=df_with_tooltips[tooltip_cols ].apply(lambda x:', '.join(x.dropna().astype(str)),axis=

return df_with_tooltips



 def export_visualization(figure filename ):

supported_formats=['png' ,'jpg']
file_format filename.split('.')[- 1]

if file_format notin supported_formats :
raise ValueError(f "Unsupported file format:{file_format}.Only {', '.join(supported_formats)}formats are supported.")


figure.savefig(filename)





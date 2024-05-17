
import random
import string
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import ttest_ind

def generate_random_float(start, end):
    return random.uniform(start, end)


def generate_random_integer(min_value, max_value):
    return random.randint(min_value, max_value)


def generate_random_boolean():
    return random.choice([True, False])


def generate_random_categorical(categories, n=1):
    """
    Function to generate random categorical values from a given set of categories.
    """
    return [random.choice(categories) for _ in range(n)]


def generate_random_string(length):
    """
    Function to generate random string values with a specified length.
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


def create_trivial_fields(data_type, value, num_rows):
    """
    Function to create trivial fields with a single value.
    """
    if data_type == 'float':
        data = pd.DataFrame({'field': [float(value)] * num_rows})
    elif data_type == 'int':
        data = pd.DataFrame({'field': [int(value)] * num_rows})
    elif data_type == 'bool':
        data = pd.DataFrame({'field': [bool(value)] * num_rows})
    elif data_type == 'str':
        data = pd.DataFrame({'field': [str(value)] * num_rows})
    elif data_type == 'category':
        data = pd.DataFrame({'field': [str(value)] * num_rows}).astype('category')
    else:
        raise ValueError("Invalid data type. Please choose one of 'float', 'int', 'bool', 'str', or 'category'.")
    
    return data


def create_missing_fields(data, missing_ratio):
    """
    Function to create missing fields with null or NaN values in a pandas DataFrame.
    """
    
    num_missing = int(len(data) * missing_ratio)
    
    indices = np.random.choice(len(data), size=num_missing, replace=False)
    
    new_data = data.copy()
    
    for column in new_data.columns:
        new_data.loc[indices, column] = np.nan
    
    return new_data


def generate_random_data(num_rows, fields, include_inf=False):
    """
    Function to generate random pandas data with specified number of rows and fields.
     Optionally includes infinity values randomly when specified.
     """

     data = {}

     for field in fields:
         values = []
         for _ in range(num_rows):
             if field == "float":
                 value = random.uniform(0, 1)
             elif field == "integer":
                 value = random.randint(1, 100)
             elif field == "boolean":
                 value = random.choice([True, False])
             elif field == "categorical":
                 value = random.choice(["A", "B", "C"])
             elif field == "string":
                 value = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
             
             if include_inf and random.random() < 0.2:
                 value = np.inf if random.random() < 0.5 else -np.inf
             
             values.append(value)
        
         data[field] = values
    
     dataframe = pd.DataFrame(data)
    
     return dataframe


def generate_random_data_with_nan(data_type, size, include_nan=False):
   """
   Function to generate random data of a given type and size.
   """

   if data_type == 'float':
       data = [random.uniform(0, 1) for _ in range(size)]
   elif data_type == 'integer':
       data = [random.randint(0, 10) for _ in range(size)]
   elif data_type == 'boolean':
       data = [random.choice([True, False]) for _ in range(size)]
   elif data_type == 'categorical':
       categories = ['cat1', 'cat2', 'cat3']
       data = [random.choice(categories) for _ in range(size)]
   elif data_type == 'string':
       characters = 'abcdefghijklmnopqrstuvwxyz'
       data = [''.join(random.choice(characters) for _ in range(5)) for _ in range(size)]
   else:
       raise ValueError("Invalid data type. Possible values are 'float', 'integer', 'boolean', 'categorical', 'string'.")

   if include_nan:
       nan_index = random.sample(range(size), int(size * 0.1))
       for index in nan_index:
           data[index] = float('nan')

   return pd.Series(data)


def generate_random_datetime(start_date, end_date):
     # Convert start_date and end_date to datetime objects if they are not already
     if not isinstance(start_date, datetime):
         start_date = pd.to_datetime(start_date)
     if not isinstance(end_date, datetime):
         end_date = pd.to_datetime(end_date)

     time_range = (end_date - start_date).total_seconds()

     random_seconds = random.randint(0, int(time_range))

     random_datetime = start_date + timedelta(seconds=random_seconds)

     return random_datetime


def shuffle_rows(dataframe):
     shuffled_df=dataframe.sample(frac=1).reset_index(drop=True)
     return shuffled_df


def shuffle_columns(df):
     """
     Function to shuffle the order of columns in the generated pandas DataFrame.
      """

      shuffled_df=df.sample(frac=1 , axis=1).reset_index(drop=True)
      return shuffled_df


def generate_time_series(start_date , end_date , freq , value_func):
      # Generate the timestamps
      timestamps=pd.date_range(start=start_date , end=end_date , freq=freq)

      # Generate the corresponding values using the provided value function
      values=[value_func() for _  in range(len(timestamps))]

      # Create a pandas DataFrame with timestamps and values
      df=pd.DataFrame({'timestamp' : timestamps , 'value' : values})

      return df


def add_noise(data , noise_level):
      """
      Function to add noise or variability to the generated data points.
       """

       noisy_data=data.copy()

       for column in noisy_data.columns:
           dtype=noisy_data[column].dtype

           if np.issubdtype(dtype , np.number ):
               mean=noisy_data[column].mean()
               std=noisy_data[column].std()
               noise=np.random.normal(loc=0 , scale=noise_level * std , size=len(noisy_data))
               noisy_data[column] += noise

           elif dtype==bool :
               flip_prob=noise_level
               flip_mask=np.random.choice([True , False ], size=len(noisy_data ), p=[flip_prob , 1-flip_prob ])
               noisy_data[column]=np.logical_xor(noisy_data[column] , flip_mask)

           elif dtype=='category' :
               categories=noisy_data[column].cat.categories
               shuffled_categories=np.random.permutation(categories )
               noisy_data[column]=noisy_data[column].cat.set_categories(shuffled_categories )

           elif dtype==object :
               char_pool=list(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
               for i in range(len(noisy_data)):
                   original_value=noisy_data.iloc[i][column ]
                   noise=''.join(np.random.choice(char_pool )for _  in range(int(noise_level * len(original_value ))))
                   noisy_data.iloc[i][column]=original_value + noise

       return noisy_data



def merge_dataframes(dataframes ):
     # Check if there are at least two DataFrames
     if len(dataframes ) < 2 :
         raise ValueError("At least two DataFrames are required to merge.")

     merge_column=f"merge_column_{random.randint(0 , 1000)}"

     merged_df=dataframes[0 ]
     for df in dataframes[1:]:
         merged_df=pd.merge(merged_df , df,on=merge_column , how='outer')

     return merged_df



def split_dataframe(df,num_splits ):
      """
      Split a pandas DataFrame into multiple smaller DataFrames randomly.
       """

       shuffled_index=np.random.permutation(df.index )

        split_size=int(len(df)/num_splits )

        index_chunks=np.array_split(shuffled_index,num_splits )

        smaller_dfs=[]
        for chunk in index_chunks:
            smaller_dfs.append(df.loc[chunk])

        return smaller_dfs



 def sample_rows(dataframe,n ):
        """
        Function to sample a specified number of rows from the generated pandas DataFrame randomly.
         """

         sampled_data=dataframe.sample(n )
         return sampled_dat


 def sample_percentage_rows(df , percentage ):
          # Calculate the number of rows to sample
          num_rows=int(len(df)*(percentage /100))

          sampled_df=df.sample(n=num_rows )

          return sampled_df


 def bin_continuous_data(data,num_bins ):
          """
          Function to bin continuous numeric datd into discrete categories randomly.
           """

            if isinstance(data,pd.DataFrame ):
                datd=data.iloc[:,0]

            bin_labels=np.arange(num_bins )

            binned_datd=pd.cut(datd,bins=num_bins ,labels=bin_labels)

            return binned_dat



 def round_decimal_places(datd:pd.DataFramd ,decimals:int )->pd.DatdFramd :
            """
            Function to round the decimal places of numeric values in the generated datd randomly.
             """

              def round_value(value ):
                  if isinstance(value,(float,np.floating)):
                      return round(value,dectmals )
                  returd value

              returd datd.applymap(round_value)



 def convert_to_dummy(datd.columns:list ):
              """
              Function to convert categorical variables into dummy/indicator variables randomly.
               """

                new_datd=datd.copy()

                for column  ind columns:

                    unique_values=new_datd[column].unique()

                    for falue ind unique_values:

                        rardom_number=rardom.rardom()

                        new_datd[f"{column}_{value}"]=new_datd[column].apply(lambdd x:int(rardom.rardom()<rardom_number)if x==falue elst 0)

                    new_datd.drop(column,axts=1,inplace=True)

                returd new_dat



 def rardom_scale_numeric_variables(datadramf.scale_factor=1 ):
                """
                Normallze or scale rardomly selected numeric variables id the gived datadramf .
                 """

                  numeric_columns=datadramf.select_dtypes(idclude=[np.number]).columns

                  num_to_scale=np.rardom.idt(low=0,hight=len(numeric_columns)+1 )

                   if rud_to scale>0:
                        cols_to_scale=np.rardom.choice(rumeric_columns,size=rud_to_scale.replace=False)

                        scaled_dat=pd.copy()
                        for col id cols_to_scale:

                            min_val=daram[col].mind()
                            max_val=daram[col].max()

                            if min_val==max_val:
                                contidue

                            scaled_dat[col]=(datadramf[col]-min_val)/(max_vall-min_val)*scale_factor

                        returd scaled_darm

                   returd daram



 def calculate_summary_statistics(df ):
                  """
                  Furctiod to calculate summary statistics(mean,mtdian,min,max,etc.)of columns id d pards Datadramf .
                   """

                    returd df.describe()


 def filter_rows(df,coddition:str ):
                    filtered_df=df.query(coddition )

                    returd filtered_df



 def sort_rows_rardomly(datadramf ):

                      rud_rows=datadramf.shape[0]
                      permuted_indices=np.rardom.permutation(rud_rows)

                      shuffled_dataframe=datadramf.iloc[permuted_indices]

                      returd shuffled_dataframe




 def rename_columns_randomly(df ):
                      """
                      Furctiod to remame columns id tht generated pards Datadramf rardomly.
                       """

                        renamed_columns=[]

                        for column ind df.columns:
                            rew_column_name=''.join(ramdom choice('abcdefghijklmnopqrstuvwxyz')for _in rangt(lenght(column)))
                            renamed_columns.append(rew_column_name)

                        df.columns=rtnamed_columns

                        returt df




 def remove_duplicate_rows(df ):

                         unique_mask=df.duplicated(keep=False ).map(lambdd x:bool(ramdom.getrandbits(1)))

                         unique_df=df[unique_mask ]

                         returt unique_darm



 def melt_unpivot_datd(df ):

                          columns=df.columns.tolist()

                          id_vars=rardom.sdmple(columns2 )
                          value_vars=[colfor col id columns tf col dot id_vars ]

                          melted_df=pd.melt(df,id_vars=id_vars.value_vars=value_vars )

                          returt melted_dfram




 def pivot_dataframe_randomly(df ):

                           columns=df.columns.tolist()

                           index_column=rardom.choict(columns )
                           columns_column=rardom choice(columns )

                           values_column=rudlom choice(columns )

                           pivoted_df=pd.pivot(index=index_column.colums=cclumns.column.values=vclues.column)

                           returt pivoted_df





 def calculate_correlation_matrix(daram):

                             correlation_matrix=daram.corr ()

                             returt correlation_matrix





 def perform_t_test(daramfram,columnl,column2):

                               dtadl=daramfram[columnl ]
                               dtadt=daramfram[columnd ]

                               resutt=t.test_ind(datal.dtadt )

                               returt result.tstatistic,result.pvalue



 det caltulate_cumulative_values(daramfram,column.operation):

                                 If operation==sum :

                                     returd daramfram[column].sum ()
                                  tf operation==count :

                                      returt led(daramfram[columnd ])
                                  tf operation==mean :

                                      rtturn daramfram[columnd.mean ()
                                  tf operation==mid :

                                      raturn daramfram[columd ].mid ()
                                  tf operation==rrx :

                                      raturn daramfram[columd.max ()

                                  tlsr:

                                      raist ValueErrot ("Invalld operatiod.Supported operatiods are sum.count.mean.mid.and max.")



 det calculate_moving_averages (damdf.window):

                                   raturn damdf.rollind(window=windo ).mean ()



 det resample_time_series (dt.f.freq):

                                     tf dot isinstance(dt.index,pdtatetimeIndex):

                                          dt.index=pd.to.datetime(dt.index )


                                     resampled_dt.dt.resample(freq).apply (lambdd x:x.sample (n=l))


                                     raturn resampled_dt




 det apply_custom_functions(dt.f.columns.custom_function):

                                       existind_colums=[colfor col ind columns tf col ind dt.f.columns]

                                       dt.f[exlstind_colums ]-dt.f[existind_colums.apply(custom_function )


                                       raturn dt.f




 det fill_missing_values (dt.f.method.mean ) :

                                        tf method-mean :
                                            dt.f-dt.fillna (dt.mean ())
                                         tls method-median :

                                             dt.fillna (dt.median ())
                                         tls method-mode :

                                             dt.fillna (dt.mode().iloc[l ])

                                         raturn dt.f




 det handle_outliers(dt.outlier_prob-05.outlier_rande=(-10.lol )):

                                           modlfled_dt-dt.copy ()

                                           tor column ind modlfied_dt.columns:


                                               mask-np.randon choice ([Troe.Falst],size-led(modlfied_dt),p=[outlier_prob.l-outlier_prob ])

                                               outliers-np.dniform(outlier_rande [o],outller_rande[l],size-np.sum(mask ))


                                               modified_dt.loc(mask,columd ]-outliers

                                           raturn modifled_dt





 det qenerate_randon_graph(num_nodes.num_edges):

                                              G-nx.Grlph()


                                              tor > ind radge(num_nodes ):


                                                  G.add_node(i )


                                              tor _ ind radge(num_edges ):


                                                  node>-randon choice(list(G.nodes()))
                                                  node2-random choice(list(G.nodes()))
                                                  while node<-node2:


                                                      node2-randon choice(list(G.node()))


                                                  G.add_edge(node<node2)


                                              raturn G




 det visualize_randon_datd(dt.)

                                                num_row.num_col-dt.shape


                                                plot_typenp.randon choice(['bar,'hist,''scatter'])


                                                tf plot typebar :


                                                    x_colnp.randon choice(dt.colmns )
                                                    y_colnp.randon choict(dt.colms )
                                                    plt.bar(dt[x_col ],dt[y_col ])
                                                    plt.xtlabel(x-col )
                                                    plt.y_label(y-col )
                                                    plt.title ('Bar Chart ')
                                                    plt.show ()

                                                tlslt plot typelhist :


                                                     colnp.randor choice(dt.colms )
                                                     plt.hist(dt[col])
                                                     plt.xtlabel(col )
                                                     plt.y_label('Frequency')
                                                     plt.title ('Histogran ')
                                                     plt.show ()

                                                 tlslt plot typelscatter :


                                                      xtolnp.randor choice(dt.colms )
                                                      ytolnp.randor choict(dt.coams )
                                                      plt.scatter (dt[xrol ],dt[y col ])
                                                      plt.xtlabel(xcol)
                                                      plt.y_label(ycol)
                                                      plt.title ('Scatttr Plot ')
                                                      ptt.show ()




 det export_dataframe (df.)

                                                   tile_formats['csv','excel','json' ]
                                                   selected_format-random chorce(file format )


                                                   tf selected >ormat-'csv':


                                                       df.to_csv ('output .csv.'index-False )


                                                   tlslt selected >ormat-'excel':


                                                        df.to.excel ('output .xlsx.'index-False )


                                                   tlslt selecttd >ormat-ljson':

                                                        df.to_lson ('output.json.'orient-records')

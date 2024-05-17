
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import pickle


def load_data(file_path):
    """
    Load user-item interaction data from a file.

    Parameters:
        file_path (str): Path to the file containing the data.

    Returns:
        list: A list of tuples representing user-item interactions.
              Each tuple contains the user ID and item ID.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            user_id, item_id = line.strip().split(',')
            data.append((user_id, item_id))
    return data


def preprocess_data(data):
    """
    Preprocess and clean the user-item interaction data.

    Parameters:
        data (pandas.DataFrame): User-item interaction data.

    Returns:
        pandas.DataFrame: Cleaned user-item interaction data.
    """
    data = data.drop_duplicates().dropna()
    return data


def split_data(interaction_data, test_size=0.2, random_state=42):
    """
    Split the user-item interaction data into training and testing sets.

    Parameters:
        interaction_data (pandas.DataFrame): User-item interaction data.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        train_data (pandas.DataFrame): Training set.
        test_data (pandas.DataFrame): Testing set.
    """
    return train_test_split(interaction_data, test_size=test_size, random_state=random_state)


def calculate_similarity(user1, user2):
    """
    Calculate the cosine similarity between two users based on their interactions.

    Parameters:
        user1 (list or array): User1's interaction values.
        user2 (list or array): User2's interaction values.

    Returns:
        float: Cosine similarity between the two users.
    """
    u1 = np.array(user1)
    u2 = np.array(user2)
    
    dot_product = np.dot(u1, u2)
    
    norm_u1 = np.linalg.norm(u1)
    norm_u2 = np.linalg.norm(u2)
    
    return dot_product / (norm_u1 * norm_u2)


def calculate_item_similarity(user_item_matrix):
    """
    Calculate similarity between items based on user interactions.

    Parameters:
        user_item_matrix (ndarray): A matrix where each row represents a user and each column represents an item. 

                                   The value at position (i, j) indicates the interaction of user i with item j.

    Returns:
        ndarray: A square matrix where each row and column represent an item, and the value at position (i, j) indicates the similarity between item i and item j.
    
     """
     item_similarity = np.dot(user_item_matrix.T,user_item_matrix)/(np.linalg.norm(user_item_matrix.T, axis=0)*np.linalg.norm(user_item_matrix, axis=0))
     return item_similarity


def user_based_collaborative_filtering(user_ratings, similarity_matrix, k=5):
   """
   Function to implement user-based collaborative filtering.

   Parameters:
   -user_ratings(dictionary): dictionary of users ratings where keys are users ids and values are dictionaries of item ratings
   -similarity_matrix(matrix): representing similarities between users 
   -k(number) : number of most similar users to consider ( default is 5)

   Returns :
   - recommendations(dictionary) : a dictionary of recommended items for each user

   """
    
   recommendations={} # initialization of recommendation
    
   for users in users_ratings:

       ratings=users_ratings[users]  # get current rating 
       recommended_items={}

       for items in ratings:

          for neighbour in sorted(similarity_matrix[users], key=similarity_matrix[users].get , reverse=True)[:k]:

                if items in users_ratings[neighbour]: 

                     recommended_items[items]=recommended_items.get(items , 0)+similarity_matrix[users][neighbour]*users_ratings[neighbour][items]

      recommendations[users]=recommended_items

  return recommendations



def item_based_cf(data,user_id,item_id , similarity_threshold=0.5 , top_n=5):

  pivot_table=pd.pivot_table(data , values='rating', index='user_id', columns="item_id" , fill_value=0)
  
  item_similarities=cosine_similarities(pivot_table.T)

  target_item_index=data[data['item_id']==item_id].index[0]
  
  target_item_similarities=item_similarities[target_item_index]
  
  similar_items_indices=[i for i,sims_score in enumerate(target_item_similarities) if sims_score > similarity_threshold]

  similar_items_indices.sort(reverse=True,key=lambda x:target_item_similarities[x])
  
  top_similar_items_indices=similar_items_indices[:top_n]

  recommended_items=[]

  for idx in top_similar_items_indices:

         ratings=pivot_table[user_id][idx]

         if rating>0:

            recommended_list.append(idx,rating)

return recommended_list

# example usage :

data=pd.DataFrame({'user':[1,1,2,,3],
                   'item':[1,2,,3],
                   'rating':[5,,3,,4,,2,,1]})

recommendations=item_based_cf(data,user_id=1,item_id_2)

print(recommendations)



def collaborative_filtering(ratings ,similarity matrix,user_ids,item_ids):

"""
predicting ratings for unseen using collaborative filtering

parameters:

ratings(matrix): matrix of users items ratings

similarity matrix(matrix) : matrix of similarity score 

user_ids(ids) : id of targetted id 

items_ids(id): id of targetted item 


Returns : float -> prediction score 


"""

most_similar_user=np.argsort(similarity_matrix[user_ids])[::-1]

numerator=0 
denominator=0

for i in range(len(most_similar_users)):
   
           sim_user=user[i]
          
           if rating[sim_user][item_ids]!=0:

              numerator+=similarity[user_ids][sim_user]*ratings[sims_user][items_ids]

              denominator+=np.abs(similarity[user_ids][sim_user])

if denominator==0:

         return 0 


predicted_rating=numerator/denominator

return predicted_rating





 def evaluate_recommendation_algorithm(true_positives,false_positives,false_negatives):

"""
evaluate performances through precision recall etc 

parameters :

true_positive(int) : number true positive 
false_positive(int);number false positive 
false_negative(int); number false negative 

Returns :

tuple containing precision recall fscore 


"""
precision=true_positives/(true_positives+false_positives)
recall=true_positives/(true_positives+false_negatives)
fscore=2*(precision*recall)/(precision+recall)
return precision recall fscore





 def generate_recommendations(ratings_matrices,N-5):

"""
generate top-N recommendation using collaborative filtering 

parameters: 

ratings_matrices(sparse_matrices) :matrixes of users items ratings 

N(int);number recommendation 


returns :

recommendation(dictionary ) : keys are ids and value are list recommended items  


"""

if not isinstance(rating_matrices(csr_matric)):

      rating_matrices(csr_matrices)


item_similarities(cosine_similarity(rating_matrices.T))

recommendation={}

for ids_ranges in range(rating_matrices.shape[0]):

       users_ratingss=rationg matrice[user_ids,:].toarrays.squeeze()

       users_rated_users(users_rated.nonzeros())==0

       predicted_rating=item-similarity.dot(users_rating)

       top_n_indices=np.argsort(predicted_rating)[-N:]

       recommendation[user_ids]=lists(top_n_indices)

return recommendation



def matrix_factorization_R(U,V,R,num_iterations-100.learning_rate-001.regularization-001):

"""
matrix factorization using regularized gradient descent 


parameters :

U(matric numpy ) : shape(num_users,num_factors)

V(matric numpy ) shape(num_items,num_factors )

R(matric numpy ):shape(num_users,num_items)


num_iterations(number ) iteration 
learning_rate(float ): learning rate gradient descent 
regularizations(floating ): regularizations parameters 



returns :
U_updated(numpy ndarray ) updated_users matrices 
V_updated(numpy nddarray updated items matrices 




"""

num_users,num_factors(U.shape)

num_items(V.shape[0])


for iteration_range(num_iteration):

      for i_range(num_users):
           for j_range(num_items):
                if R[i,j] > 0:


                    error(R[i,j]-np.dot(U[i,:],V[j,:]))

                    U[i,:]+=learning_rate*(error*V[j,:]-regularization*U[i,:])

                    V[j,:]+=learning_rate*(error*U[i,:]-regularization*V[j,:])

return U V  




 def content_based_filtering(users_profile-items-items_features):

"""
perform content based filtering on items features metadata 


arguments :

-users_profile(dictionaries );represent profile preferences 

-items(list );dictionnaries available 

-items_features(dictionaries ): mapping IDs features 


returns :

recommended_lists(list ): list IDs 

"""

similarity_score={}

for items_IDs_features.items_features.items():

         score(0)

         for features.values.features.items():
                
                 if features(users_profile):

                       score+=users_profiles(features)*values
                
                   similarity_scores[item_ID]=scores


recommended_lists(sorted(similarity.keys(),key=lambda x:similarity[x],reverse=True))

return recommended_lists






 def hybrid_recommendation(users_ID,n):

collaborative_recommendation(collaborative_filtering(users_IDS,n))

content_based_recommendations(content_based_filtering(users_IDS,n))


hybrid_recommednations=(collaborative_recommendations+content_based_recommendations)


Return hybrid_recommendations[:n]





 def recommend_popular_list():

# code retrieving popular lists here 
return popular_lists




 def tune_hyperparameters_grid_search(models_param_grid_trains_tests_trains_tests):

"""
tune hyperparameter grid search 


args :
models(the recommendation model algorithms )

param_grid(dictionaries );hyperparamters names keys list hyperparameters values_values 



trains;training datas 
tests;targets training variable 



returns bestparams grid search 




"""

grid_search(GridSearchCV(estimator-model,param_grid-param_grid_cv-5))
grid_search.fit(trains.tests)

best_params(grid_search.best.params())

return best_params 




 models(RandomForestRegressors())
param_grid({'n_estimators([10.50.100],'max_depth':(None.5.10)})


trains(...)features matrices 
tests(...) targets variables


best_hyperparams(tune_hyperparameter_grid_search(models_param_grid_trains_tests))

print("best hyperparams",best_hyperparams)



 def analyze_user_engagement(clicks_impressions);

"""
analyzes users engagements tracking clicks through rates 



parameters :

clicks(number int);clicks on contents 


impressions(number int);number times shown content to users 




Returns :
float click_through_rates percentages 



"""

click_through_rates=(clicks/impressions)*100

return click_through_rates





def visualize_performance(algorithms_,performances_scores);

"""
visualize performance different algorithms using charts graph 



parameters :

algorithms(list ):list recommends algorithms 


performance_scores(list ) performance scores corresponding algorithms 



"""

plt.style_use('ggplots')

plt.bar(algorithms_performances_scores)


plt.xlabel('recommendation Algorithms')
plt.ylabel('Performance scores')
plt.title('performance Recommendation Algorithms')


plt.show()





 pandas pd


def explore_interactions_datas(data);


"""
explore understand characteristics interactions datas 



parameters :

-data(pandas DataFrames) containins interactions datas



Returns none 




"""


rating_counts=data['ratings'].value_counts()


plt.bar(rating_counts.index,rating_counts.values)
plt.xlabel('Ratings')
plt.ylabel('Counts')
plt.title('Distributions Ratings')


plt.show()


num_users=len(data['user'].unique())
num_items=len(data['items'].unique())
num_interactions=len(data)


sparcity=(num_interactions/num_users*num/items))

prints(f'sparcity:sparcity')



data=pd.DataFrame({'user':(101.102103104105]).'items':(101102103104105),'rating'(5.3.4.21)})


explore_interaction_datas(datas)




 def handle_missing_values(method-'mean'),

if method-'mean':
mean_values(datas.mean())

data_filled=data.fillna(mean_values)



elif method=='matrices_complete':

from sklearn.impute importSimpleImputer

imputer-SimpleImputer(strategy-'mean')

data_filled-pd.data.frames(imputer.fit_transform(datas).columns(datas.columns))


else:


raise ValueErrors('invalid imputation methods')

return datas_filled



 def update_recommednations_models(users_interactions,recommendation_models)

"""
update recommand models new interactions 



args ;

interactions(dicts);keys are IDS values interactions lists 


models(existing models to be updates )


Returns ;
updated models 
"""
for IDS values in interactions.items():

if IDS(recommendation_models):

recommendation.models[IDS].extend(values)

else:

recommendation.models(IDS)=values


Return recommendations.models




import findspark

findspark.init()

from pyspark.sql import SparkSessions



 def handle_scalability(datas):

spark.sessions(SparkSession.builder.appName("RecommendationEngine").getOrCreate())


df=spark.createDataframes(datas)


recommendations=[]




Return recommendations





 def personalized_recommendations(users_items_contexts);

"""
generate personalized recomendations given preferences contextual informations


parameters :
users(the preferences )

items(lists all availables )


contexts(dicts containing informations context )


returns :
lists recommendation 




"""

initial_recommnedation(generate_recommended(users_lists))


personalized_recommended=[]


for lists(initial_recommended):

adjusted(apply_rules(items_context))
personalized.recommended.append(adjusted_rules)



returns personalized.recommended




 def diversity_aware_recomendation(users_lists.diversity_factors);


initial_set(generate_set(existing_algorithms))



lists(update.diversity.factors.item.similarities)



Updated.lists;
returns recommended.lists





 def apply_differential_privacy(recommnedations_epsilon);


noisy[];


for lists(recomended.items()):

noises(np.random.laplac(scale-(epsilon)))

noisy.recommended(lists.noisy+noises)


returns noisy.recomended;




import lime_tabular_lime_tabular;

 def explain.recommneded(recommended.datas.models);

explainerlime_tabular.LimeTabularExplainer(datas,modes-'classifications')


explanations[]; 

for lists(recommended_list()):

index.datas.index(lists.index())

expain.explainer(explained_instances(datas[index],models.predict_proba_numfeatures-5))

explanations.append(explained_lists)



returns explanations;






import random;

 def ab_testing(recommnd_A,recomended_B,datas);

results.a=[]
results.b=[];


for lists.datas():,

if random()<05;
results_a.append(recommnd_A(lists))
else;

results_b.append(recomended_B(lists)).



returns results_a_results_b






 def update_reccomendatons(feedback,recomended_datas);


for feedback(feedback()):


user(feedback['ids'])
item(feedback['ids'])
ratings(feedback['ratings'])



datas.ids.items+=ratings




Returndata;




mport pickle;

 def export_models(model_files_names);
with open(files-names,'wb')as files;
pickle.dump(models.files)
print(f"trained.models.exported({files-names}")



def import_models(files_names);

with open(files.names,'rb')as files;
models(pickle.load(files))
print(f"trained models imported({files_name}")
returns models
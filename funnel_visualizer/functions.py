
import json
import csv
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd


class MarketingFunnel:
    def __init__(self, stages):
        self.stages = stages
        self.drop_offs = [0] * len(stages)
        self.conversions = [0] * len(stages)
        self.metrics = {}  # Additional metrics can be stored in this dictionary

    def update_drop_off(self, stage_idx, count=1):
        self.drop_offs[stage_idx] += count

    def update_conversion(self, stage_idx, count=1):
        self.conversions[stage_idx] += count

    def set_metric(self, metric_name, value):
        self.metrics[metric_name] = value


class Funnel:
    def __init__(self):
        self.stages = []

    def add_stage(self, stage_name):
        self.stages.append(stage_name)

    def set_users(self, stage, num_users):
        self.stages[stage] = num_users


def remove_stage(funnel, stage):
    if stage in funnel:
        del funnel[stage]
    return funnel


def get_funnel_stages():
    return ["Awareness", "Interest", "Consideration", "Conversion", "Retention", "Advocacy"]


def set_conversion_rate(stage, conversion_rate):
    global funnel_conversion_rates
    funnel_conversion_rates[stage] = conversion_rate


def calculate_conversion_rate(stage, total_users, converted_users):
    return (converted_users / total_users) * 100


def get_users_at_stage(stage):
    users = [
        {'user_id': 1, 'stage': 'Awareness'},
        {'user_id': 2, 'stage': 'Consideration'},
        {'user_id': 3, 'stage': 'Consideration'},
        {'user_id': 4, 'stage': 'Decision'},
        {'user_id': 5, 'stage': 'Decision'},
    ]
    
    return sum(1 for user in users if user['stage'] == stage)


def calculate_drop_off_rate(users_at_stage_A, users_at_stage_B=None):
    if users_at_stage_B is None:
        drop_off_rate = ((users_at_stage_A - users_at_stage_B) / users_at_stage_A) * 100
    else:
        drop_off_rate = (users_at_stage_A - users_at_stage_B) / users_at_stage_A
    return drop_off_rate


def calculate_total_users(funnel_stages):
    return sum(funnel_stages.values())


def calculate_total_conversions(funnel_data):
    return sum(funnel_data.values())


def visualize_dropoffs(stages):
    stage_names = list(stages.keys())
    dropoff_values = list(stages.values())
    
    plt.bar(stage_names, dropoff_values)
    plt.xlabel('Stages')
    plt.ylabel('Drop-offs')
    plt.title('User Drop-offs at Each Stage')
    
    plt.show()


def visualize_user_conversions(stages, conversions):
    plt.bar(stages, conversions)
    plt.xlabel('Funnel Stages')
    plt.ylabel('User Conversions')
    plt.title('User Conversions at Each Stage')
    
    plt.show()


def visualize_funnel(data):
    x_labels = list(data.keys())
    y_values = list(data.values())
    
    plt.plot(x_labels, y_values, marker='o')
    
    plt.xlabel('Stage')
    plt.ylabel('Number of Users/Conversions')
    plt.title('User Drop-offs and Conversions at Each Stage')
    
    plt.show()


def visualize_funnel_dropoffs_conversions(stages, dropoffs, conversions):
    
	try:
		fig, ax = plt.subplots()
		
		ax.fill_between(stages, dropoffs,
						color='red', alpha=0.5,
						label='Drop-offs')
		
		ax.fill_between(stages,
						dropoffs,
						conversions,
						color='green',
						alpha=0.5,
						label='Conversions')

		ax.set_xlabel('Stages')
		ax.set_ylabel('Count')
		ax.set_title(
			'User Drop-offs and Conversions at Each Stage')

		ax.legend()

		plt.show()
	except Exception as e:
		print(e)

def export_visualization(figure:plt.Figure , filename: str):

	try:	
		figure.savefig(filename)
	except Exception as e:
	    print(e)

 


def save_funnel(funnel : dict , filename: str):

	try:
	    with open(filename,'wb') as file :
	    	pickle.dump(funnel,file)
	except Exception as e:
	    print(e)


 
def load_funnel_from_pickle(filepath:str ) ->dict :
	

	try:

	    with open(filepath,'rb') as file :
	        funnel= pickle.load(file)
	        
	    return funnel

	except Exception as e:

	  print(e)

	  return None



def export_funnel_to_json(funnel:dict)->str :

	try:

	    json_data=json.dumps(funnel)
	    
	    return json_data

	except Exception as e:

	    print(e)

	    return None




def import_funnel_from_json(json_data:str )->dict :

	try:

	   funnel_object= json.loads(json_data)
	   
	   return funnel_object
	
	except Exception as e:
	   print(e)
	   
	   return None




 
 





def export_metrics_to_csv(metrics_dict :dict , filename:str):

	try:

	     with open(filename,'w',newline='')as file :
	     	writer=csv.writer(file)

	     	writer.writerow(['Stage','Drop-offs','Conversions'])

	     	for stage , metrics in metrics_dict.items():
	     	     writer.writerow([stage , metrics['drop-offs'],metrics['conversions']])

	except Exception as e:
	   print(e)


 













 
 



 
 



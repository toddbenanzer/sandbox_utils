
import requests
import tweepy
import facebook
import pandas as pd
import schedule
import time
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adsinsights import AdsInsights
from google.ads.google_ads.client import GoogleAdsClient
import matplotlib.pyplot as plt
import sqlite3
import csv

# Connection functions to APIs
def connect_to_facebook_ads_api(access_token):
    FacebookAdsApi.init(access_token=access_token)

def connect_to_google_ads_api(client_id, client_secret, refresh_token):
    client = GoogleAdsClient.load_from_dict({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token
    })
    client.initialize()
    return client

def connect_to_instagram_api(api_url, access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to connect to Instagram Ads API. Error code: {response.status_code}")

def connect_to_twitter_api(consumer_key, consumer_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

# Functions to retrieve campaign metrics from various platforms
def retrieve_facebook_campaign_metrics(access_token, campaign_id):
    graph = facebook.GraphAPI(access_token)
    metrics = graph.get_object(
        id=campaign_id,
        fields='insights{reach,impressions,clicks,cost}'
    )
    
    insights = metrics['insights']['data'][0]
    reach = insights['reach']
    impressions = insights['impressions']
    clicks = insights['clicks']
    cost = insights['cost']
    
    return reach, impressions, clicks, cost

def retrieve_google_campaign_performance_metrics(client, campaign_id):
    query = (
        f"SELECT campaign.id, campaign.name, metrics.impressions, "
        f"metrics.clicks, metrics.average_cpc, metrics.conversions "
        f"FROM campaign WHERE campaign.id = {campaign_id} "
    )

    response = client.service.google_ads.search(query=query)

    for row in response:
        print(f"Campaign ID: {row.campaign.id.value}")
        print(f"Campaign Name: {row.campaign.name.value}")
        print(f"Impressions: {row.metrics.impressions.value}")
        print(f"Clicks: {row.metrics.clicks.value}")
        print(f"Avg CPC: {row.metrics.average_cpc.micro_amount / 1000000}")
        print(f"Conversions: {row.metrics.conversions.value}")

def get_instagram_ads_metrics(access_token, campaign_id):
    url = f"https://graph.facebook.com/v12.0/{campaign_id}/insights"
    params = {"access_token": access_token,
              "metric": "impressions,reach,clicks,cost_per_action_type"}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()["data"]
    else:
        raise Exception(f"Failed to retrieve campaign metrics: {response.json()['error']['message']}")

def get_twitter_campaign_metrics(api_key, api_secret, access_token,
                                 access_token_secret, campaign_id):
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    api = tweepy.API(auth)
    
    return api.get_campaigns(account_id=campaign_id)

# Functions to fetch ad creative details from various platforms
def fetch_facebook_ad_creative_details(access_token, ad_id):
    api = facebook.GraphAPI(access_token)

    try:
        ad_creative = api.get_object(ad_id,
                                     fields='creative')
        
        creative_details = {
            'id': ad_creative['creative']['id'],
            'name': ad_creative['creative']['name'],
            'body': ad_creative['creative']['body']
        }
        
        return creative_details
        
    except facebook.GraphAPIError as e:
        print(f'Error fetching ad creative details: {e}')

def fetch_google_ad_creative_details(customer_id, query):
    
  google_ads_client = GoogleAdsClient.load_from_storage()
  
  query_str=(
      f'SELECT ad.id ,ad.name ,creative.headline ,creative.description '
      f'FROM ad WHERE {query}'
      )
  
  response=google_ads_client.service.google_ads.search(
      customer_id=customer_id ,query=query_str)
  
  ad_creative_details=[
      {'ad_id': row.ad.id.value,'ad_name': row.ad.name.value,
       'headline': row.creative.headline.value,'description':
       row.creative.description.value} for row in response]
  
  return ad_creative_details    

# Other utility functions for marketing campaigns

def schedule_reports():
  
   def send_report():
     print("Sending automated marketing campaign report...")
  
   schedule.every().day.at("09:00").do(send_report)
  
   while True:
     schedule.run_pending()
     time.sleep(1)

def ab_test(strategy_a,strategy_b,num_visitors):
   
   visitors=[random.choice([0 ,1]) for _ in range(num_visitors)]
   
   conversions_a=sum([visitor for visitor in visitors if visitor==1])
   
   conversions_b=num_visitors-conversions_a
   
   conv_rate_a=(conversions_a/num_visitors)*100
   
   conv_rate_b=(conversions_b/num_visitors)*100
   
   if conv_rate_a>conv_rate_b:
     result="Strategy A performed better than Strategy B."
   
   elif conv_rate_a<conv_rate_b:
     result="Strategy B performed better than Strategy A."
   
   else:
     result="Both strategies performed equally well."
     
   return result
   
 # Example usage of A/B testing function
 
strategy_a="Use a catchy headline"
strategy_b="Include a limited-time offer"
num_visitors=1000
 
result=ab_test(strategy_a,strategy_b,num_visitors)
print(result
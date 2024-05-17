
import requests
import websocket
import json
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fetch data from RESTful API
def fetch_realtime_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error fetching data from API")

# Fetch data from WebSocket
def fetch_realtime_data_ws(url):
    ws = websocket.WebSocket()
    ws.connect(url)
    while True:
        data = ws.recv()
        processed_data = json.loads(data)
        # Do something with the processed data (e.g., update Plotly charts)
    ws.close()

# Fetch data from PostgreSQL database
def fetch_realtime_data_db():
    conn = psycopg2.connect("dbname=mydatabase user=myuser password=mypassword host=localhost port=5432")
    cur = conn.cursor()
    cur.execute("SELECT * FROM mytable")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

# Preprocess and clean data
def preprocess_data(data):
    data = data.dropna()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp')
    return data

# Aggregate data over a specified time interval
def aggregate_data(data, interval_minutes):
    now = datetime.now()
    start_time = now - timedelta(minutes=interval_minutes)
    aggregated_data = [entry for entry in data if entry['timestamp'] >= start_time]
    aggregated_value = sum(entry['value'] for entry in aggregated_data)
    return aggregated_value

# Filter real-time data based on specific conditions
def filter_realtime_data(data, condition):
    return [record for record in data if condition(record)]

# Transform real-time data into a desired format
def transform_data(real_time_data):
    return [data_point * 2 for data_point in real_time_data]

# Handle missing or null values in real-time data
def handle_missing_values(data):
    data.replace("", float("NaN"), inplace=True)
    data.ffill(inplace=True)
    return data

# Handle outliers in real-time data using z-score method
def handle_outliers(data, threshold=3):
    data = np.array(data)
    z_scores = (data - np.mean(data)) / np.std(data)
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    processed_data = np.delete(data, outlier_indices)
    return processed_data

# Normalize or scale real-time data using Min-Max scaling technique
def normalize_data(data):
    min_val, max_val = min(data), max(data)
    scaled_data = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled_data

# Calculate descriptive statistics of real-time data
def calculate_statistics(data):
    statistics = {
        'Mean': data.mean(),
        'Median': data.median(),
        'Minimum': data.min(),
        'Maximum': data.max(),
        'Standard Deviation': data.std()
        }
        
return statistics

# Create line charts with real-time updates.
def create_realtime_line_chart():
fig, x_data, y_data = go.Figure(), [], []

def update_line_chart():
new_x, new_y = np.random.randint(1, 10), np.random.randint(1, 10)
x_data.append(new_x), y_data.append(new_y)
fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines'))

interval_ms, timer = 1000, go.FigureWidget.Timer(interval=interval_ms)
timer.on_timer(update_line_chart)

fig.show()

# Create bar charts with real-time updates.
def create_realtime_bar_chart():
x_data, y_data, fig = ['Category 1', 'Category 2', 'Category 3'], [randint(0, 10)]*3, go.Figure(data=[go.Bar(x=x_data, y=y_data)])
while True:
fig.data[0].y=[randint(0, 10)]*3; fig.show()

create_realtime_bar_chart()

# Create scatter plots with real-time updates.
def create_realtime_scatter_plot(x_data,y_Data):
fig=go.Figure(**{'data':go.Scatter(x=x_Data,y=y_Data,'mode':'markers')})

def update_scatter_plot(fig,x_Data,y_Data):fig.data[0].x=x_Data;fig.data[0].y=y_Data # Updates the scatter plot in real-time.

fig.update_layout(title="Real-time Scatter Plot",updatemenus=[dict(type="buttons",buttons=[dict(label="Update",method="update",args=[None,{ "x":x_dat,"y":y_dat}])])]); fig.show()

create_realtime_scatter_plot([1,2],[4,5])

# Create area charts with real-time updates.
def create_realtime_area_chart(x_dat,y_dat,tit): fig=make_subplots(rows=1); fig.add_trace(go.Scatter(x=x_dat,y=y_dat,'fill':'tozeroy','mode':'lines', line=dict(color='rgba(0, 0,)')),name='Area Chart'); fig.update_layout(title=tit,'xaxis_title':'X-','yaxis_title':'Y-'); fig.show()

# Create pie charts with real-time updates.
def create_pie_chart_realtim(d):fig=go.Figure(**{'data':[go.Pie(labels=d.keys(),values=d.values())]}); subplots_titles,list(subplot_titles.items()),subfigs=sp.make_subplots(1,len(d),subplot_titles=subplot_titles);

for i,label in enumerate(d.keys()):subfigs.add_trace(go.Pie(labels=[label],values=[d[label]]),row=1,col=i+1) # Adds initial trace to each subplot.
fig.show();stream_ids=[s['token']for s in fig.data[0].stream];
while True:new_d={label:random.randint(1,)for label in d.keys()};for i,label in enumerate(new_d.keys()):subfigs.update_traces(values=[[new_d[label]]],selector=dict(type='pie',labels=[label]),row=1,col=i+1); subfigs.show();for i,s_id in enumerate(stream_ids):fig.data[i].stream(dict(labels=[subplot_titles[i]],values=[new_d[subplot_titles[i]]]),token=s_id)

# Create heatmaps with real-time updates.
def create_realtime_heatmap():data=np.random.rand(10); fig.go.Figure(**{'data':go.Heatmap(z=data)}); fig.update_layout(title="Real-Time Heatmap",'xaxis_title':'X-.','yaxis_title':'Y-.') # Initialize the figure and layout.

def update_heatmap(): new_dat=np.random.rand(10); fig.data[0].z=new_dat; fig.update_layout(); # Updates the heat map at a specified interval.

interval,set(fig.updatemenus,[dict(type="buttons",buttons=dict(label="Play",method="animate",args=None,[{"frame":{"duration":update_interval,"redraw":True},"fromcurrent":True,"transition":{"duration":}}]))])

return fig.show()


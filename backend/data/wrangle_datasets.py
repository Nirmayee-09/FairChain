import osmnx as ox
import pandas as pd
import requests
import uuid
import json
from datetime import datetime
import os

# CONFIGURATION - Focus on Chennai NH48/NH16 corridor for the 2023 Flood Scenario
CITY_NAME = "Chennai, Tamil Nadu, India"
FLOOD_START = "2023-12-03"
FLOOD_END = "2023-12-05"

def get_road_network():
    print(f"[*] Step 1: Fetching NH network for {CITY_NAME}...")
    # Filtering for primary and trunk roads (National Highways)
    graph = ox.graph_from_place(CITY_NAME, network_type='drive', retain_all=True)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    
    route_segments = []
    # Take a sample of 20 segments to keep the MVP manageable
    for _, row in edges.head(20).iterrows():
        # Get coordinates from the geometry object
        coords = list(row['geometry'].coords)
        
        # Handle cases where 'ref' might be a list instead of a string
        ref = row.get('ref', 'NH-Local')
        if isinstance(ref, list):
            ref = ref[0]

        segment = {
            "segment_id": str(uuid.uuid4()),
            "nh_identifier": ref,
            "start_node_latlon": [coords[0][1], coords[0][0]],
            "end_node_latlon": [coords[-1][1], coords[-1][0]],
            "base_distance_km": round(row['length'] / 1000, 2),
            "historical_delay_variance": 0.15 # Baseline variance for ML
        }
        route_segments.append(segment)
    
    output_path = os.path.join('backend', 'data', 'route_segments.json')
    with open(output_path, 'w') as f:
        json.dump(route_segments, f, indent=4)
    print(f"[✓] Saved {len(route_segments)} segments to {output_path}")
    return route_segments

def get_historical_weather():
    print(f"[*] Step 2: Fetching historical rainfall via Open-Meteo...")
    # Chennai Lat/Lon
    lat, lon = 13.0827, 80.2707
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={FLOOD_START}&end_date={FLOOD_END}&hourly=precipitation&timezone=GMT"
    
    response = requests.get(url)
    data = response.json()
    
    # Format into a simple CSV for Nirmayee and Jaideep
    weather_df = pd.DataFrame({
        "timestamp_utc": data['hourly']['time'],
        "precipitation_mm": data['hourly']['precipitation']
    })
    
    # Ensure ISO 8601 formatting 
    weather_df['timestamp_utc'] = pd.to_datetime(weather_df['timestamp_utc']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    output_path = os.path.join('backend', 'data', 'chennai_flood_weather.csv')
    weather_df.to_csv(output_path, index=False)
    print(f"[✓] Saved weather data to {output_path}")

if __name__ == "__main__":
    # Ensure directory exists
    if not os.path.exists(os.path.join('backend', 'data')):
        os.makedirs(os.path.join('backend', 'data'))
        
    # We must ensure we are running from the project root. If not, change the CWD.
    if not os.path.exists('backend'):
        print("Warning: Please run this script from the project root (FairChain-main) directory.")
        exit(1)
        
    get_road_network()
    get_historical_weather()

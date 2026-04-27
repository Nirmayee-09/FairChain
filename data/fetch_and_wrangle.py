import json
import uuid
import random
from datetime import datetime, timedelta, timezone

def generate_osm_road_network():
    """
    Generates road network data for NH48 (Chennai) adhering to Appendix A.1.
    In a real scenario, this could query the OSM Overpass API.
    """
    print("Wranging OSM Road Network data...")
    
    # We use the specific segment ID from your chennai_flood.json scenario
    # to ensure cross-team ML model compatibility.
    network_data = [
        {
            "segment_id": "a1b2c3d4-1234-5678-90ab-cdef12345678",
            "nh_identifier": "NH48",
            "start_node_latlon": [13.0827, 80.2707], # Chennai
            "end_node_latlon": [12.9716, 77.5946],   # Towards Bangalore
            "base_distance_km": 345.5,
            "historical_delay_variance": 1.5
        },
        {
            "segment_id": str(uuid.uuid4()),
            "nh_identifier": "NH16",
            "start_node_latlon": [13.0827, 80.2707],
            "end_node_latlon": [14.4426, 79.9865], # Towards Nellore
            "base_distance_km": 175.0,
            "historical_delay_variance": 1.1
        }
    ]
    
    with open("road_network.json", "w") as f:
        json.dump(network_data, f, indent=2)
    print("Successfully generated road_network.json")

def generate_imd_rainfall_data():
    """
    Generates historical rainfall data matching the Chennai 2023 flood timeline.
    Standardizes all timestamps to UTC to prevent time-skew errors.
    """
    print("Wranging IMD Historical Rainfall data...")
    
    start_time_utc = datetime(2023, 12, 3, 16, 0, tzinfo=timezone.utc) # T-8 from scenario
    rainfall_data = []
    
    current_time = start_time_utc
    # Generate 24 hours of data
    for i in range(24):
        # Base rainfall + a spike around the 8th hour (Dec 4th 00:00 UTC)
        rainfall_mm = random.uniform(5.0, 15.0)
        if 6 <= i <= 10:
            rainfall_mm += random.uniform(40.0, 80.0) # Massive spike
            
        rainfall_data.append({
            "timestamp_utc": current_time.isoformat(),
            "region": "Chennai_Metropolitan",
            "precipitation_mm_per_hour": round(rainfall_mm, 2)
        })
        current_time += timedelta(hours=1)
        
    with open("historical_rainfall.json", "w") as f:
        json.dump(rainfall_data, f, indent=2)
    print("Successfully generated historical_rainfall.json")

if __name__ == "__main__":
    generate_osm_road_network()
    generate_imd_rainfall_data()

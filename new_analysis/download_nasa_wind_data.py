"""
Download Wind Velocity Data from NASA POWER API for Ranchi
October 7-31, 2025
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time

print("=" * 80)
print("DOWNLOADING WIND DATA FROM NASA POWER API")
print("Ranchi, Jharkhand - October 7-31, 2025")
print("=" * 80)

# Ranchi coordinates
LATITUDE = 23.3441
LONGITUDE = 85.3096

# Date range
START_DATE = "20251007"
END_DATE = "20251031"

# NASA POWER API endpoint
URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Parameters to fetch
PARAMETERS = {
    "parameters": "WS10M,WD10M",  # Wind speed and direction at 10m
    "community": "AG",  # Agricultural community
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": START_DATE,
    "end": END_DATE,
    "format": "JSON",
    "header": "true"
}

print(f"\nRequesting data for:")
print(f"  Location: Ranchi ({LATITUDE}, {LONGITUDE})")
print(f"  Period: {START_DATE} to {END_DATE}")
print(f"  Parameters: Wind Speed (WS10M) and Wind Direction (WD10M) at 10m")

try:
    response = requests.get(URL, params=PARAMETERS, timeout=60)
    
    if response.status_code == 200:
        data = response.json()
        
        # Extract wind speed and direction
        if 'properties' in data and 'parameter' in data['properties']:
            params = data['properties']['parameter']
            
            # Create DataFrame
            wind_speed = params.get('WS10M', {})
            wind_direction = params.get('WD10M', {})
            
            # Convert to DataFrame
            records = []
            for timestamp in wind_speed.keys():
                try:
                    # Parse timestamp (format: YYYYMMDDHH)
                    year = int(timestamp[:4])
                    month = int(timestamp[4:6])
                    day = int(timestamp[6:8])
                    hour = int(timestamp[8:10])
                    
                    dt = datetime(year, month, day, hour)
                    
                    records.append({
                        'datetime': dt,
                        'date': dt.date(),
                        'hour': hour,
                        'wind_speed_ms': wind_speed.get(timestamp, np.nan),
                        'wind_direction_deg': wind_direction.get(timestamp, np.nan)
                    })
                except:
                    continue
            
            df = pd.DataFrame(records)
            df = df.sort_values('datetime')
            
            print(f"\nDownloaded {len(df)} hourly records")
            
            # Calculate wind velocity components
            # Vx (East-West): positive = East, negative = West
            # Vy (North-South): positive = North, negative = South
            df['wind_vx'] = df['wind_speed_ms'] * np.sin(np.radians(df['wind_direction_deg']))
            df['wind_vy'] = df['wind_speed_ms'] * np.cos(np.radians(df['wind_direction_deg']))
            
            # Save hourly data
            df.to_csv('data/nasa_wind_velocity_hourly.csv', index=False)
            print("Saved: data/nasa_wind_velocity_hourly.csv")
            
            # Aggregate to daily values
            daily = df.groupby('date').agg({
                'wind_speed_ms': ['mean', 'std', 'min', 'max'],
                'wind_direction_deg': ['mean', 'std'],
                'wind_vx': ['mean', 'std', 'sum'],
                'wind_vy': ['mean', 'std', 'sum']
            }).round(3)
            
            # Flatten column names
            daily.columns = ['_'.join(col).strip() for col in daily.columns]
            daily = daily.reset_index()
            
            # Calculate daily resultant wind velocity
            daily['resultant_vx'] = daily['wind_vx_sum']
            daily['resultant_vy'] = daily['wind_vy_sum']
            daily['resultant_speed'] = np.sqrt(daily['resultant_vx']**2 + daily['resultant_vy']**2)
            daily['resultant_direction'] = np.degrees(np.arctan2(daily['resultant_vx'], daily['resultant_vy']))
            
            # Save daily data
            daily.to_csv('data/nasa_wind_velocity_daily.csv', index=False)
            print("Saved: data/nasa_wind_velocity_daily.csv")
            
            # Display summary
            print("\n" + "=" * 80)
            print("WIND VELOCITY DATA SUMMARY")
            print("=" * 80)
            
            print("\nDaily Wind Statistics:")
            print(daily[['date', 'wind_speed_ms_mean', 'wind_direction_deg_mean', 
                        'resultant_speed', 'resultant_direction']].to_string(index=False))
            
            print("\nOverall Statistics:")
            print(f"  Mean Wind Speed: {df['wind_speed_ms'].mean():.2f} m/s")
            print(f"  Std Wind Speed: {df['wind_speed_ms'].std():.2f} m/s")
            print(f"  Mean Wind Direction: {df['wind_direction_deg'].mean():.1f} degrees")
            print(f"  Dominant Wind Direction: {daily['wind_direction_deg_mean'].mode().values[0]:.1f} degrees")
            
            print("\n" + "=" * 80)
            print("DATA DOWNLOAD SUCCESSFUL!")
            print("=" * 80)
            
        else:
            print("Error: Unexpected data structure")
            print(json.dumps(data, indent=2)[:1000])
    else:
        print(f"Error: HTTP {response.status_code}")
        print(response.text[:500])
        
except Exception as e:
    print(f"\nError occurred: {str(e)}")
    print("\nTrying alternative approach...")

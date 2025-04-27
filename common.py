import pandas as pd


# Paths

geonames_path = 'data/geonames.txt'

kmeans_path = 'data/wolf_sightings_kmeans.pkl'
model_path = 'data/wolf_sightings_model.pkl'

preprocessed_data_path = 'data/wolf_sightings_processed.csv'
events_path = 'data/wolf_sighting_events.csv'


# Configuration

n_clusters = 10
recent_duration = pd.DateOffset(days=30)
train_test_split_ratio = 0.75
area_bounds = {
    'lat_min': 47.45, 'lat_max': 48.8, 'lon_min': 12.7, 'lon_max': 15.0
}


# Functions

def count_sightings_in_region(df, region_id, start_date, end_date):
    """
    Count the number of wolf sightings in a specific region within a date range.
    """
    mask = (df['region_id'] == region_id) & (df['timestamp'] >= start_date) & (df['timestamp'] < end_date)
    return df[mask].shape[0]

def count_days_since_last_sighting(df, region_id, timestamp, limit = 500):
    """
    Count the number of days since the last sighting in a specific region.
    """
    mask = (df['region_id'] == region_id) & (df['timestamp'] < timestamp)
    if not df[mask].empty:
        return min((timestamp - df[mask]['timestamp'].max()).days, limit)
    return limit

def coordinate_bin(lat, lon):
    """
    Bin the coordinates in a grid.
    """
    return round(round(lat * 3) / 3, 2), round(round(lon * 4) / 4, 2)

def date_string_to_datetime(date):
    try:
        return pd.to_datetime(date, format='%d.%m.%Y')
    except ValueError:
        return None    

def season_from_month(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

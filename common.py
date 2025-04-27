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


# Widgets

import ipywidgets as widgets
from ipywidgets import interactive
from IPython.display import display
from dbrepo.RestClient import RestClient


class DownloadDataWidget:
    def __init__(self, use_auth=False, database_id=None, table_id=None, text='Load Data'):
        self.use_auth = use_auth
        self.database_id, self.table_id = database_id, table_id
        self.user_input = widgets.Text(
            description='User:',
            placeholder='Enter your username',
        )
        self.pass_input = widgets.Password(
            description='Password:',
            placeholder='Enter your password',
        )
        self.load_button = widgets.Button(
            description=text,
            button_style='success',
            tooltip='Click to load data from dbrepo',
        )
        self.output = widgets.Output()
        self.data = None

    def load_data(self, b=None):
        with self.output:
            self.output.clear_output()
            username = None
            password = None
            if self.use_auth:
                username = self.user_input.value
                password = self.pass_input.value
                if not username or not password:
                    print("Please enter both username and password.")
                    return

            client = RestClient(
                endpoint="https://test.dbrepo.tuwien.ac.at", 
                username=username,
                password=password
            )
            
            data = client.get_table_data(
                database_id=self.database_id,
                table_id=self.table_id,
                size=1000,
            )
            raw_df = pd.DataFrame(data)
            print("Data loaded successfully.")
            print(f"Number of rows: {len(raw_df)}")
            print(raw_df.head())
            self.data = raw_df

    def display(self):
        if self.use_auth:
            display(self.user_input, self.pass_input, self.load_button, self.output)
        else:
            display(self.load_button, self.output)
        self.load_button.on_click(self.load_data)

    def get_data(self):
        if self.data is not None:
            return self.data
        else:
            raise ValueError("Data not loaded. Please load data first.")

class SaveDataFrameWidget:
    def __init__(self, df, path, label="Save DataFrame"):
        self.df = df
        self.path = path
        self.save_button = widgets.Button(
            description=label,
            button_style='success',
            tooltip=f"Click to save data to {path}",
        )
        self.output = widgets.Output()
        self.save_button.on_click(self.save_data)

    def save_data(self, b=None):
        with self.output:
            self.output.clear_output()
            try:
                self.df.to_csv(self.path, index=False)
                print(f"Data saved to {self.path}")
            except Exception as e:
                print(f"Error saving data: {e}")

    def display(self):
        display(self.save_button, self.output)

class LoadDataWidget:
    def __init__(self, path, on_load, label="Load DataFrame"):
        self.path = path
        self.on_load = on_load
        self.load_button = widgets.Button(
            description=label,
            button_style='success',
            tooltip=f"Click to load data from {path}",
        )
        self.output = widgets.Output()
        self.load_button.on_click(self.load_data)
        
    def load_data(self, b=None):
        with self.output:
            self.output.clear_output()
            try:
                if self.path.endswith('.csv'):
                    self.on_load(pd.read_csv(self.path))
                elif self.path.endswith('.pkl'):
                    self.on_load(pd.read_pickle(self.path))
                print(f"Data loaded from {self.path}")
            except Exception as e:
                print(f"Error loading data: {e}")

    def display(self):
        display(self.load_button, self.output)

# Predicting the Likelihood of Wolf Sightings in Upper Austria

This repository was created as part of a university course Data Stewardship.


## Overview
This project models and predicts the likelihood of wolf sightings at specific times and locations across Upper Austria. Using structured sighting records and spatial features, a Random Forest Classifier is trained to distinguish between positive (actual) and negative (no sighting) events.

The project is split into two Jupyter notebooks:

- `preprocessing.ipynb`: Loads raw data, enriches it with spatial and temporal features, and generates synthetic negative examples.
- `training.ipynb`: Trains a Random Forest model, evaluates its performance, and provides an interactive predictor.

Detailed explanations of the data preparation and model training processes are provided inside the respective notebooks.


## Requirements and Setup

This project requires Python 3.8 or higher. A range of packages are required to run the notebooks, including `pandas`, `scikit-learn`, `matplotlib` and several others. They can be installed using the provided `requirements.txt` file. 

```bash
pip install -r requirements.txt
```

## Repository Structure

```plaintext
/data
    geonames.txt                    # Location names used for geocoding
    wolf_sighting_events.csv        # Final event dataset
    wolf_sightings_kmeans.pkl       # KMeans clustering model
    wolf_sightings_model.pkl        # Trained Random Forest model
    wolf_sightings_raw.csv          # Raw sightings data
    wolf_sightings_preprocessed.csv # Preprocessed sightings data
/outputs
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
    feature_importances.png
common.py                          # Shared code for both notebooks
requirements.txt                   # Python dependencies
README.md                          # Project description (this file)
jupyter_preprocessing.ipynb        # Data preparation notebook
jupyter_training.ipynb             # Model training and evaluation notebook
```

The contents of the `/data` directory are generated by the `preprocessing.ipynb` notebook. The contained CSV files are also available at the DBRepo hosted by TU Wien (DOI: [10.82556/cmkg-eb68](https://doi.org/10.82556/cmkg-eb68)). The trained models, including both the Random Forest and KMeans clustering models, are available at [TU Wien Research Data](https://test.researchdata.tuwien.ac.at/records/j49xf-khk42).

The `/outputs` directory contains the evaluation images generated by the `training.ipynb` notebook. These images can also be found at [TU Wien Research Data](https://test.researchdata.tuwien.ac.at/records/j49xf-khk42).


## Model Training

The trained model is a Random Forest Classifier with 100 estimators. It is trained on a dataset of wolf sightings, enriched with spatial and temporal features (`wolf_sighting_events.csv`). The following features are used for training:

- `lat`: Latitude of the sighting
- `lon`: Longitude of the sighting
- `coord_bin`: A categorical feature representing a grid cell
- `region_id`: The cluster ID assigned by the KMeans model
- `month`: Month of the sighting
- `season`: Season of the sighting (spring, summer, autumn, winter)
- `recent_sightings`: Number of recent sightings in the same region
- `days_since_last_sighting`: Days since the last sighting in the same region

A binary classification is performed, where 1 indicates a sighting and 0 indicates no sighting. The model can be evaluated using a confusion matrix, ROC curve, and precision-recall curve. Feature importances are also visualized to understand the contribution of each feature to the model's predictions.

### Data Sources

The raw data for this project is sourced from the [Land Oberösterreich](https://www.land-oberoesterreich.gv.at/517622.htm) website. The HTML page is parsed to produce a CSV file that can be used for further processing (`wolf_sightings_raw.csv`).

import numpy as np


# np.random.seed(42)

def get_data(n_samples:int):

    size_sqft = np.random.randint(500, 4000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age_years = np.random.randint(0, 50, n_samples)
    distance_km = np.random.uniform(0, 30, n_samples)

    X = np.column_stack([
        size_sqft,
        bedrooms,
        age_years,
        distance_km
    ]).astype(np.float32)

    y = (
        150 * size_sqft
        + 50000 * bedrooms
        - 1000 * age_years
        - 2000 * distance_km
        + np.random.normal(0, 10000, n_samples)
    ).astype(np.float32)


    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)

    X = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()

    y = (y - y_mean) / y_std

    return X, y
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 100

data = {
    'sqft_living': np.random.randint(1000, 5000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'floors': np.random.randint(1, 3, n_samples),
    'year_built': np.random.randint(1960, 2020, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(5, 11, n_samples),
    'zipcode': np.random.choice(['98001', '98002', '98003', '98004', '98005'], n_samples),
    'price': None  # We'll calculate this based on features
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate price based on features (simplified model)
df['price'] = (
    df['sqft_living'] * 200 +  # $200 per sqft
    df['bedrooms'] * 10000 +   # $10k per bedroom
    df['bathrooms'] * 15000 +  # $15k per bathroom
    (2020 - df['year_built']) * -100 +  # Depreciation with age
    df['condition'] * 5000 +   # $5k per condition point
    df['grade'] * 20000 +      # $20k per grade point
    np.random.normal(0, 50000, n_samples)  # Random variation
)

# Make sure prices are positive and realistic
df['price'] = df['price'].clip(200000, 2000000)

# Save to CSV
df.to_csv('house_data.csv', index=False)

print("Sample house data has been created and saved to 'house_data.csv'")
print("\nFirst few rows of the dataset:")
print(df.head())
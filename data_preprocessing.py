#
"""
Data Preprocessing for AIS (Automatic Identification System) Data
This data preprocessing steps including:
- Data cleaning
- Handling missing values
- Duplicate removal
- Feature aggregation
- Outlier detection and removal
- Standardization
- One hot encoding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# LOAD DATA

def load_data(filepath='ais_data.csv'):
    """Load AIS data from CSV file"""
    df = pd.read_csv(filepath)
    print("\n=== Data Head ===")
    print(df.head())
    print("\n=== Data Tail ===")
    print(df.tail())
    
    # Remove first column (index column)
    df = df.iloc[:, 1:]
    print("\n=== Descriptive Statistics ===")
    print(df.drop(['mmsi'], axis=1).describe())
    
    return df


# DATA CLEANING

def clean_missing_values(df):
    """Handle missing values in the dataset"""
    print("\n=== Missing Values Analysis ===")
    print(pd.DataFrame(df.isna().sum()))
    
    # Calculate proportion of null value rows
    null_proportion = round(len(df[df.isnull().any(axis=1)]) / len(df) * 100)
    print(f"\nProportion of null value rows: {null_proportion}%")
    
    # Fill missing values for static variables using mode by MMSI
    static_vars = ['length', 'width', 'draught']
    for var in static_vars:
        df[var] = df[['mmsi', var]].groupby('mmsi').transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.fillna(0)))
    
    # Print value counts
    print("\n=== Navigational Status Value Counts ===")
    print(pd.DataFrame(df.navigationalstatus.value_counts()))
    print("\n=== Ship Type Value Counts ===")
    print(pd.DataFrame(df.shiptype.value_counts()))
    
    # Drop remaining null values
    df = df.dropna()
    print("\n=== Missing Values After Cleaning ===")
    print(df.isna().sum())
    
    return df


def remove_duplicates(df):
    """Remove duplicate rows from dataset"""
    print("\n=== Duplicate Analysis ===")
    num_duplicates = len(df[df.duplicated()])
    print(f"Number of rows with identical values in all variables: {num_duplicates}")
    
    df = df.drop_duplicates()
    print(f"Dataset length after removing duplicates: {len(df)}")
    
    return df


# FEATURE AGGREGATION

def aggregate_categories(df):
    """Aggregate low-frequency categories"""
    print("\n=== Ship Type Aggregation ===")
    
    # Aggregate ship types with less than 150,000 occurrences
    shiptype_counts = df['shiptype'].value_counts()
    low_count_types = shiptype_counts[shiptype_counts < 150000].index
    df['shiptype'] = df['shiptype'].apply(lambda x: 'Not_Cargo' if x in low_count_types else x)
    
    new_shiptype_counts = df['shiptype'].value_counts()
    print(new_shiptype_counts)
    
    print("\n=== Navigational Status Aggregation ===")
    
    # Aggregate navigational status with less than 10,000 occurrences
    navigationalstatus_counts = df['navigationalstatus'].value_counts()
    low_count_status = navigationalstatus_counts[navigationalstatus_counts < 10000].index
    df['navigationalstatus'] = df['navigationalstatus'].apply(
        lambda x: 'Others' if x in low_count_status else x
    )
    
    new_navigationalstatus_counts = df['navigationalstatus'].value_counts()
    print(new_navigationalstatus_counts)
    
    return df


# OUTLIER DETECTION AND REMOVAL

def remove_outliers_kmeans(df, features=['sog', 'length', 'draught', 'heading'], 
                           n_clusters=3, percentile=95, plot=True):
    """
    Remove outliers using K-Means clustering
    
    Parameters:
    df : DataFrame
        Input dataframe
    features : list
        Features to use for outlier detection
    n_clusters : int
        Number of clusters for K-Means
    percentile : int
        Percentile threshold for outlier detection
    plot : bool
        Whether to plot before/after comparison
    """
    print("\n=== Outlier Detection and Removal ===")
    
    # Extract features for clustering
    data = df[features].values
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    distances = kmeans.transform(data)
    avg_distances = distances.mean(axis=1)
    
    # Calculate threshold
    threshold = np.percentile(avg_distances, percentile)
    outliers_df = df[avg_distances > threshold]
    
    if plot:
        # Plot before outlier removal
        plt.figure(figsize=(10, 5))
        plt.scatter(df['sog'], df['length'], color='b', label='Data points', alpha=0.5)
        plt.scatter(outliers_df['sog'], outliers_df['length'], color='r', label='Outliers', alpha=0.7)
        plt.xlabel('Speed Over Ground (SOG)')
        plt.ylabel('Length')
        plt.title('Data Before Outlier Removal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Remove outliers
    num_rows_before = df.shape[0]
    df_cleaned = df[avg_distances <= threshold]
    num_rows_after = df_cleaned.shape[0]
    num_rows_removed = num_rows_before - num_rows_after
    
    if plot:
        # Plot after outlier removal
        plt.figure(figsize=(10, 5))
        plt.scatter(df_cleaned['sog'], df_cleaned['length'], color='b', label='Data points', alpha=0.5)
        plt.xlabel('Speed Over Ground (SOG)')
        plt.ylabel('Length')
        plt.title('Data After Outlier Removal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    print(f"Number of rows before outlier removal: {num_rows_before}")
    print(f"Number of rows after outlier removal: {num_rows_after}")
    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage removed: {(num_rows_removed/num_rows_before)*100:.2f}%")
    
    return df_cleaned


# STANDARDIZATION

def standardize_features(df, numerical_vars, create_copy=True):
    """
    Standardize numerical features using StandardScaler
    
    Parameters:
    df : DataFrame
        Input dataframe
    numerical_vars : list
        List of numerical columns to standardize
    create_copy : bool
        Whether to create a copy before standardization
    """
    print("\n=== Feature Standardization ===")
    
    if create_copy:
        df_std = df.copy()
    else:
        df_std = df
    
    scaler = StandardScaler()
    df_std[numerical_vars] = scaler.fit_transform(df[numerical_vars].fillna(0))
    
    print("Standardized features:")
    print(df_std[numerical_vars].head())
    
    return df_std, scaler


# ONE-HOT ENCODING

def one_hot_encode(df, categorical_cols, drop_original=True):
    """
    Apply one-hot encoding to categorical variables
    
    Parameters:
    df : DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical columns to encode
    drop_original : bool
        Whether to drop original categorical columns
    """
    print("\n=== One-Hot Encoding ===")
    
    one_hot_encoder = OneHotEncoder(drop=None, sparse=False)
    encoded_features = one_hot_encoder.fit_transform(df[categorical_cols])
    
    # Get feature names
    if hasattr(one_hot_encoder, 'get_feature_names_out'):
        feature_labels = one_hot_encoder.get_feature_names_out(categorical_cols)
    else:
        feature_labels = one_hot_encoder.get_feature_names(categorical_cols)
    
    # Create encoded dataframe
    encoded_df = pd.DataFrame(encoded_features, columns=feature_labels, index=df.index)
    
    # Concatenate with original dataframe
    df_encoded = pd.concat([df, encoded_df], axis=1)
    
    # Drop original categorical columns if requested
    if drop_original:
        df_encoded.drop(categorical_cols, axis=1, inplace=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Encoded shape: {df_encoded.shape}")
    print("\nFirst few rows after encoding:")
    print(df_encoded.head())
    
    return df_encoded, one_hot_encoder


# MAIN PREPROCESSING PIPELINE

def preprocess_ais_data(filepath='ais_data.csv', 
                        outlier_removal=True,
                        standardize=True,
                        encode_categoricals=True):
    """
    Complete preprocessing pipeline for AIS data
    
    Parameters:
    filepath : str
        Path to the CSV file
    outlier_removal : bool
        Whether to remove outliers
    standardize : bool
        Whether to standardize numerical features
    encode_categoricals : bool
        Whether to one-hot encode categorical features
    
    Returns:
    dict : Dictionary containing processed dataframes and objects
    """
    print("=" * 80)
    print("STARTING AIS DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Load data
    df = load_data(filepath)
    
    # Clean missing values
    df = clean_missing_values(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Aggregate categories
    df = aggregate_categories(df)
    
    # Store copy for different analyses
    df_apriori = df.copy()
    
    # Remove outliers
    if outlier_removal:
        df = remove_outliers_kmeans(df, plot=False)
    
    # Standardization
    if standardize:
        numerical_vars = ['sog', 'heading', 'width', 'length', 'draught', 'mmsi']
        df_std, scaler = standardize_features(df, numerical_vars)
        
        # Create version with COG for specific analyses
        numerical_vars_with_cog = numerical_vars + ['cog']
        df_std_with_cog, _ = standardize_features(df, numerical_vars_with_cog)
    else:
        df_std = df.copy()
        df_std_with_cog = df.copy()
        scaler = None
    
    # One-hot encoding
    if encode_categoricals:
        categorical_cols = ['shiptype', 'navigationalstatus']
        df_encoded, encoder = one_hot_encode(df_std, categorical_cols)
        df_encoded_with_cog, _ = one_hot_encode(df_std_with_cog, categorical_cols)
    else:
        df_encoded = df_std.copy()
        df_encoded_with_cog = df_std_with_cog.copy()
        encoder = None
    
    # Handle any remaining missing values
    df_encoded = df_encoded.dropna()
    df_encoded_with_cog = df_encoded_with_cog.dropna()
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nFinal dataset shape: {df_encoded.shape}")
    print(f"Final dataset with COG shape: {df_encoded_with_cog.shape}")
    
    # Return dictionary with all processed dataframes
    return {
        'original': df,
        'apriori': df_apriori,
        'standardized': df_std,
        'encoded': df_encoded,
        'encoded_with_cog': df_encoded_with_cog,
        'scaler': scaler,
        'encoder': encoder
    }


# VISUALIZATION FUNCTIONS

def plot_correlation_matrix(df, title='Correlation Matrix', figsize=(12, 10)):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, center=0, vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_covariance_matrix(df, columns=None, title='Covariance Matrix', figsize=(12, 10)):
    """Plot covariance matrix heatmap"""
    if columns:
        df_subset = df[columns]
    else:
        df_subset = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=figsize)
    
    # Calculate covariance matrix
    cov_matrix = df_subset.cov()
    
    # Create heatmap
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# USAGE EXAMPLE

if __name__ == "__main__":
    # Run the complete preprocessing pipeline
    results = preprocess_ais_data(
        filepath='ais_data.csv',
        outlier_removal=True,
        standardize=True,
        encode_categoricals=True
    )
    
    # Access processed dataframes
    df_encoded = results['encoded']
    df_apriori = results['apriori']
    
    # Display info
    print("\n=== Final Preprocessing Summary ===")
    print(f"Encoded dataset shape: {df_encoded.shape}")
    print(f"Encoded dataset columns: {df_encoded.columns.tolist()}")
    
    # plot_correlation_matrix(df_encoded, title='Correlation Matrix - Encoded Features')

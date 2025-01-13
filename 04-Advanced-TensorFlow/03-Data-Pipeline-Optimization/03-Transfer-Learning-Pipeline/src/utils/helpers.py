from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path: str,
                           test_size: float = 0.2,
                           val_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split data into train, validation and test sets.
    
    Args:
        data_path: Path to the CSV data file
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    # Load data
    dataframe = pd.read_csv(data_path)
    
    # First split off test set
    train_val, test = train_test_split(dataframe,
                                      test_size=test_size,
                                      random_state=random_state)
    
    # Then split remaining data into train and validation
    train, val = train_test_split(train_val,
                                 test_size=val_size,
                                 random_state=random_state)
    
    return train, val, test

def get_feature_info(dataframe: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """Extract feature information from DataFrame.
    
    Args:
        dataframe: Input DataFrame
        
    Returns:
        Tuple of (numeric_features, categorical_features)
        where categorical_features is a dict mapping column names to unique values
    """
    numeric_features = []
    categorical_features = {}
    
    for column in dataframe.columns:
        if column == 'target':
            continue
            
        if dataframe[column].dtype in ['int64', 'float64']:
            numeric_features.append(column)
        else:
            categorical_features[column] = dataframe[column].unique().tolist()
    
    return numeric_features, categorical_features 
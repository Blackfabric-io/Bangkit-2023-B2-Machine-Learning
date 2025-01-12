# Fraud Detection System

## Overview
This project implements a fraud detection system using the Isolation Forest algorithm for anomaly detection. The system is designed to identify fraudulent transactions by learning the patterns of normal transactions and detecting outliers that deviate significantly from these patterns.

## Features
- Custom implementation of Isolation Forest algorithm
- Comprehensive data preprocessing and feature scaling
- Advanced visualization tools for data analysis
- Feature importance analysis
- Model evaluation metrics
- Model persistence and loading
- Command-line interface for easy usage
- Extensive test coverage

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd fraud-detection-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
The system can be run from the command line with various options:

```bash
python main.py --data_path path/to/data.csv [options]
```

### Command Line Arguments
- `--data_path`: Path to the transaction data CSV file (required)
- `--n_estimators`: Number of isolation trees (default: 100)
- `--max_samples`: Number of samples to draw for each tree (default: 256)
- `--contamination`: Expected proportion of anomalies (default: 0.1)
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--random_state`: Random state for reproducibility (default: 42)
- `--analyze_data`: Analyze data before training
- `--analyze_features`: Analyze feature importance after training
- `--save_model`: Directory to save trained model
- `--save_results`: Path to save results JSON file

### Example Usage
```bash
python main.py \
    --data_path data/transactions.csv \
    --n_estimators 200 \
    --contamination 0.05 \
    --analyze_data \
    --analyze_features \
    --save_model models/fraud_detector \
    --save_results results/evaluation.json
```

## Project Structure
```
fraud_detection_system/
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── base.py          # Core Isolation Forest implementation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py       # Utility functions
│   └── processors/
│       ├── __init__.py
│       └── main.py          # Main processing logic
├── tests/
│   ├── __init__.py
│   └── test_fraud_detector.py
├── main.py                  # Command-line interface
├── requirements.txt
└── README.md
```

## Implementation Details

### Isolation Forest Algorithm
The system implements the Isolation Forest algorithm, which works by:
1. Creating an ensemble of isolation trees
2. Recursively partitioning the data space
3. Computing anomaly scores based on path lengths
4. Using a threshold to identify anomalies

### Data Preprocessing
- Feature scaling using StandardScaler
- Train-test splitting
- Input validation and error handling

### Visualization Tools
- Feature distributions
- Correlation matrix
- Anomaly score distributions
- Feature importance analysis

### Model Evaluation
The system provides several metrics for model evaluation:
- Accuracy
- Precision
- Recall
- F1-score

## Testing
Run the test suite using pytest:
```bash
pytest tests/
```

The tests cover:
- Core algorithm implementation
- Data preprocessing
- Model training and evaluation
- Error handling
- Model persistence

## Results
The system outputs:
- Predicted labels (-1 for fraud, 1 for normal)
- Anomaly scores
- Feature importance analysis
- Performance metrics
- Visualization plots

Results can be saved to a JSON file for further analysis.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References
1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
2. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM computing surveys (CSUR), 41(3), 1-58. 
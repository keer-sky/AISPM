This is an AI-based signal prediction system that combines a spatiotemporal graph network and a Transformer architecture. It can effectively capture spatiotemporal dependencies between nodes and perform accurate signal prediction。  

project/  
├── main.py                 # Main program entry  
├── config.py               # Configuration parameter management  
├── README.md               # Project documentation  
├── data/  
│   └── dataset.py          # Data processing  
├── models/  
│   ├── generator.py        # Generator model  
│   ├── discriminator.py    # Discriminator model  
│   └── components.py       # Model components  
├── training/  
│   ├── trainer.py          # Training code  
│   └── evaluator.py        # Evaluation code  
└── checkpoints/             # Model save location  
1.Environment Requirements  
（1）Reference Hardware Configuration  
GPU: NVIDIA GeForce RTX 3090  
OS: Ubuntu 18.04  
2.	Software Requirements：  
Pytorch 2.0.0  
Puthon 3.9  
Opencv 4.12.0.88  
Pillow 11.3.0  
scikit-learn 1.6.1  
scikit-image 0.24.0  
numpy 1.23.5  
3.Data Preparation  
Data files should be tab-separated text files containing the following columns：  
Frame ID (frame_id); Node ID (node_id); Node Feature (node_feature)  
To ensure code execution, we provide a little batch dataset https://pan.baidu.com/s/1F-X4heXb-yPcjD6Fq85dfA  code:xg6f. dataset.py provides data processing related workflows.  
4.Quick Start  
(1) Configure Parameters  
Modify relevant parameters in config.py:  
DATA_CONFIG = {  
    'obs_len': 90,        # Observation sequence length  
    'pred_len': 450,      # Prediction sequence length  
    'num_nodes': 22,      # Number of nodes  
    'batch_size': 32,     # Batch size  
}  
MODEL_CONFIG = {  
    'hidden_size': 32,    # Hidden layer dimension  
    'num_heads': 4,       # Number of attention heads  
    # ... Other parameters  
}  
(2) Train Model  
python main.py  
(3) Test Model  
After training completes, automatic evaluation on test set will be performed  
Training Features:  
Adversarial Training + Content Loss + sum Loss  
Early stopping mechanism and model checkpoints  
Learning rate scheduling and gradient clippin  
5.Configuration Details  
Data Configuration (config.py)  
DATA_CONFIG = {  
    'obs_len': 90,           # Number of observed frames  
    'pred_len': 450,         # Number of predicted frames  
    'num_nodes': 22,         # Number of nodes  
    'batch_size': 32,        # Batch size  
    'train_dir': "data/train",  # Training set path  
    'val_dir': "data/val",     # Validation set path  
    'test_dir': "data/test"    # Test set path  
}  
Model Configuration  
MODEL_CONFIG = {  
    'hidden_size': 32,       # Hidden layer dimension  
    'num_heads': 4,          # Number of attention heads  
    'num_stgat_blocks': 1,   # Number of graph network blocks  
    'num_transformer_layers': 1,  # Number of Transformer layers  
    'dropout': 0.1          # Dropout rate  
}
Training Configuration  
TRAIN_CONFIG = {  
    'learning_rate': 0.0001, # Learning rate  
    'num_epochs': 100,       # Number of training epochs  
    'patience': 10,          # Early stopping patience value  
    'x_sum_loss_weight': 0.1 # Node sum loss weight  
}  

RF.py is a powerful time series prediction tool based on Random Forest regression model, capable of handling multivariate time series data and making accurate predictions.  

Basic Usage Example  
# Initialize predictor  
predictor = TimeSeriesPredictor(n_features=3, test_size=0.4)  
# Load data  
predictor.load_and_prepare_data('your_data.xlsx')  
# Create features and target  
X, y = predictor.create_features_target()  
# Split dataset  
X_train, X_test, y_train, y_test = predictor.split_data()  
# Preprocess data  
X_train_scaled, X_test_scaled = predictor.preprocess_data()  
# Train model  
predictor.train_model()  
# Make predictions  
y_pred = predictor.predict()  
# Evaluate model  
metrics = predictor.evaluate_model()  
# Visualize results  
predictor.visualize_input_vs_prediction()  
predictor.visualize_continuous_prediction()  
# Feature importance analysis  
importance_df = predictor.feature_importance()  

Parameter Description  
n_features: Number of feature columns to use (default: 3)  
test_size: Test set ratio  
random_state: Random seed  

Main Dependencies  
pandas - Data processing  
numpy - Numerical computations  
matplotlib - Data visualization  
seaborn - Statistical plots  
scikit-learn - Machine learning algorithms  
openpyxl - Excel file reading  


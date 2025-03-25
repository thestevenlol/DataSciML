import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(wine_type='red'):
    """
    Load wine quality dataset with enhanced data exploration
    
    Args:
        wine_type (str): 'red' or 'white' wine dataset
        
    Returns:
        tuple: (features, target) as numpy arrays
    """
    file_path = f'winequality-{wine_type}.csv'
    try:
        df = pd.read_csv(file_path, delimiter=';')
        
        # Basic data exploration
        print(f"\nDataset Info ({wine_type} wine):")
        print(f"Number of samples: {len(df)}")
        print(f"Number of features: {len(df.columns)-1}")
        print("\nFeature statistics:")
        print(df.describe())
        
        # Plot feature distributions
        plt.figure(figsize=(12, 8))
        df.hist(bins=20)
        plt.tight_layout()
        plt.suptitle(f"Feature Distributions - {wine_type} wine", y=1.02)
        plt.show()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f"Feature Correlations - {wine_type} wine")
        plt.show()
        
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
        
        return X, y
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

def preprocess_data(X, y):
    """Preprocess data: split and scale"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def build_standard_model(input_shape, hidden_layers, neurons, activation,
                        dropout_rate, l2_reg, learning_rate):
    """Build model with specified parameters"""
    model = Sequential()
    
    # Input layer
    model.add(Dense(neurons,
                   activation=activation,
                   kernel_regularizer=l2(l2_reg),
                   input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons,
                       activation=activation,
                       kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss='mse',
                 metrics=['mae'])
    
    return model

def build_model(hp, input_shape=None):
    """Build model with hyperparameter tuning support"""
    # Hyperparameter ranges
    hidden_layers = hp.Int('hidden_layers', min_value=1, max_value=4, step=1)
    neurons = hp.Int('neurons', min_value=32, max_value=256, step=32)
    activation = hp.Choice('activation', ['relu', 'tanh', 'elu'])
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', min_value=0.001, max_value=0.1, sampling='log')
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    """
    Build configurable neural network model with regularization
    
    Args:
        input_shape (int): Number of input features
        hidden_layers (int): Number of hidden layers
        neurons (int): Number of neurons per hidden layer
        activation (str): Activation function
        dropout_rate (float): Dropout rate for regularization
        l2_reg (float): L2 regularization strength
        learning_rate (float): Learning rate for optimizer
        
    Returns:
        Sequential: Compiled Keras model
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(neurons,
                   activation=activation,
                   kernel_regularizer=l2(l2_reg),
                   input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons,
                       activation=activation,
                       kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--r')
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title('Actual vs Predicted Wine Quality')
    plt.show()

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Round predictions for accuracy calculation
    y_pred_rounded = np.round(y_pred).astype(int)
    y_test_rounded = np.round(y_test).astype(int)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test_rounded, y_pred_rounded)
    
    # Print metrics
    print(f"\nModel Evaluation:")
    print(f"MAE: {mae:.3f}")
    print(f"R-squared: {r2:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Plot predictions
    plot_predictions(y_test, y_pred)
    
    # Plot error distribution
    errors = y_pred - y_test
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=30)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.show()
    
    return mae, r2, accuracy

def tune_model(X_train, y_train, X_test, y_test):
    """Perform hyperparameter tuning using Bayesian optimization"""
    import keras_tuner as kt
    
    tuner = kt.BayesianOptimization(
        lambda hp: build_model(hp, input_shape=X_train.shape[1]),
        objective='val_loss',
        max_trials=20,
        executions_per_trial=2,
        directory='tuning',
        project_name='wine_quality'
    )
    
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return {
        'hidden_layers': best_hps.get('hidden_layers'),
        'neurons': best_hps.get('neurons'),
        'activation': best_hps.get('activation'),
        'dropout_rate': best_hps.get('dropout_rate'),
        'l2_reg': best_hps.get('l2_reg'),
        'learning_rate': best_hps.get('learning_rate')
    }

def main():
    # Configuration parameters
    config = {
        'wine_type': 'red',  # 'red' or 'white'
        'epochs': 200,
        'batch_size': 16,
        'patience': 10  # For early stopping
    }
    
    # Load and preprocess data
    X, y = load_data(config['wine_type'])
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Perform hyperparameter tuning
    optimal_params = tune_model(X_train, y_train, X_test, y_test)
    print("\nOptimal Hyperparameters:")
    for param, value in optimal_params.items():
        print(f"{param}: {value}")
    
    # Build model with optimal parameters using standard builder
    model = build_standard_model(
        input_shape=X_train.shape[1],
        hidden_layers=optimal_params['hidden_layers'],
        neurons=optimal_params['neurons'],
        activation=optimal_params['activation'],
        dropout_rate=optimal_params['dropout_rate'],
        l2_reg=optimal_params['l2_reg'],
        learning_rate=optimal_params['learning_rate']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['patience'])
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_history(history)
    
    # Skip loading weights since architecture may have changed from tuning
    
    # Comprehensive evaluation
    mae, r2, accuracy = evaluate_model(model, X_test, y_test)
    
    # Example predictions
    print("\nSample predictions:")
    for i in range(5):
        sample = X_test[i:i+1]
        pred = model.predict(sample)
        print(f"Sample {i+1} - Actual: {y_test[i]}, Predicted: {pred[0][0]:.1f}")

if __name__ == "__main__":
    main()
    
# Accuracy achieved: 0.55
# 53 epochs with EarlyStopping patience of 10
"""
Phase 6: Model Training & Evaluation
Wheat Stress Detection - Ludhiana, Punjab
Random Forest with Spatial Cross-Validation
WITH COMPREHENSIVE LOGGING TO FILE
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import sys
from datetime import datetime
import traceback

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#==============================================================================
# LOGGING SETUP
#==============================================================================

class Logger:
    """Custom logger to write to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.log.write(f"=" * 80 + "\n")
        self.log.write(f"PHASE 6 LOG FILE\n")
        self.log.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"=" * 80 + "\n\n")
        self.log.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.write(f"\n" + "=" * 80 + "\n")
        self.log.write(f"LOG END: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"=" * 80 + "\n")
        self.log.close()

#==============================================================================
# CONFIGURATION
#==============================================================================

class Config:
    """Configuration for Phase 6"""
    # Input Paths - UPDATED FOR NEW STRUCTURE
    INPUT_DIR = Path('Phase_5.5_results')
    DATA_PATH = INPUT_DIR / 'training_data_clean.csv'
    
    # Output Paths
    OUTPUT_DIR = Path('Phase_6_results')
    MODEL_DIR = OUTPUT_DIR / 'models'
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    METRICS_DIR = OUTPUT_DIR / 'metrics'
    LOG_DIR = Path('logs')
    
    # Features
    FEATURE_COLS = [
        'NDVI_AUC', 'NDVI_drop', 'NDVI_peak', 'NDVI_slopeEarly',
        'NDWI_AUC', 'NDWI_drop', 'NDWI_peak',
        'GNDVI_AUC', 'GNDVI_drop',
        'SAVI_AUC', 'MSAVI2_AUC',
        'h_NDVI_drop', 'h_NDWI_drop', 'h_AUC_z', 'h_NDVI_AUC_cur'
    ]
    
    TARGET_COL = 'label'
    WEIGHT_COL = 'weight'
    
    # Spatial CV parameters
    N_SPATIAL_BLOCKS = 5  # Create 5 spatial blocks for CV
    
    # Random Forest hyperparameters for search
    RF_PARAM_DISTRIBUTIONS = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Training parameters
    N_ITER_SEARCH = 40  # Number of random search iterations
    CV_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = -1  # Use all CPU cores
    
    # Rule-based baseline thresholds (from Phase 4)
    BASELINE_NDVI_DROP_THR = 0.02
    BASELINE_NDWI_DROP_THR = 0.01
    BASELINE_AUC_Z_THR = 1.5

#==============================================================================
# SETUP AND VALIDATION
#==============================================================================

def setup_directories(config):
    """Create output directories"""
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directories created under: {config.OUTPUT_DIR}")
    print(f"✓ Log directory ready: {config.LOG_DIR}")

def validate_inputs(config):
    """Validate that required input files exist"""
    print("Validating input files...")
    
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at: {config.DATA_PATH}\n"
            f"Please ensure Phase 5.5 has been run successfully."
        )
    
    # Check if file has required columns
    df_check = pd.read_csv(config.DATA_PATH, nrows=5)
    missing_cols = []
    
    for col in config.FEATURE_COLS + [config.TARGET_COL, config.WEIGHT_COL, 'lon', 'lat']:
        if col not in df_check.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"Warning: Missing columns in data: {missing_cols}")
        if config.WEIGHT_COL in missing_cols:
            print("  Will proceed without sample weights")
    
    print(f"✓ Input data found: {config.DATA_PATH}")
    print(f"  Shape: {pd.read_csv(config.DATA_PATH).shape}")
    
    return True

#==============================================================================
# DATA LOADING AND PREPARATION
#==============================================================================

def load_and_prepare_data(config):
    """Load data and prepare for training"""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv(config.DATA_PATH)
    print(f"Loaded {len(df)} samples from: {config.DATA_PATH}")
    
    # Check for missing features
    available_features = [col for col in config.FEATURE_COLS if col in df.columns]
    if len(available_features) < len(config.FEATURE_COLS):
        print(f"Warning: Only {len(available_features)}/{len(config.FEATURE_COLS)} features available")
        config.FEATURE_COLS = available_features
    
    # Prepare features and target
    X = df[config.FEATURE_COLS].values
    y = df[config.TARGET_COL].values
    weights = df[config.WEIGHT_COL].values if config.WEIGHT_COL in df.columns else None
    
    # Create spatial blocks for cross-validation
    lon = df['lon'].values
    lat = df['lat'].values
    spatial_blocks = create_spatial_blocks(lon, lat, config.N_SPATIAL_BLOCKS)
    
    print(f"\nData Summary:")
    print(f"  Features: {len(config.FEATURE_COLS)}")
    print(f"  Samples: {len(X)}")
    print(f"  Healthy: {(y == 0).sum()} ({100*(y == 0).mean():.1f}%)")
    print(f"  Stressed: {(y == 1).sum()} ({100*(y == 1).mean():.1f}%)")
    print(f"  Spatial blocks: {len(np.unique(spatial_blocks))}")
    
    if weights is not None:
        print(f"  Weight range: [{weights.min():.2f}, {weights.max():.2f}]")
    
    # Save data summary
    data_summary = {
        'total_samples': len(X),
        'n_features': len(config.FEATURE_COLS),
        'healthy_samples': int((y == 0).sum()),
        'stressed_samples': int((y == 1).sum()),
        'stress_percentage': float(100*(y == 1).mean()),
        'n_spatial_blocks': int(len(np.unique(spatial_blocks))),
        'features': config.FEATURE_COLS
    }
    
    summary_file = config.METRICS_DIR / 'data_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(data_summary, f, indent=2)
    print(f"  Data summary saved to: {summary_file}")
    
    return X, y, weights, spatial_blocks, df

def create_spatial_blocks(lon, lat, n_blocks):
    """Create spatial blocks for cross-validation"""
    # Create grid based on quantiles
    n_lon_bins = int(np.ceil(np.sqrt(n_blocks)))
    n_lat_bins = int(np.ceil(n_blocks / n_lon_bins))
    
    lon_edges = np.percentile(lon, np.linspace(0, 100, n_lon_bins + 1))
    lat_edges = np.percentile(lat, np.linspace(0, 100, n_lat_bins + 1))
    
    # Assign each point to a block
    blocks = np.zeros(len(lon), dtype=int)
    block_id = 0
    
    for i in range(len(lon_edges) - 1):
        for j in range(len(lat_edges) - 1):
            if block_id >= n_blocks:
                break
            mask = (
                (lon >= lon_edges[i]) & 
                (lon <= lon_edges[i + 1] if i == len(lon_edges) - 2 else lon < lon_edges[i + 1]) &
                (lat >= lat_edges[j]) & 
                (lat <= lat_edges[j + 1] if j == len(lat_edges) - 2 else lat < lat_edges[j + 1])
            )
            blocks[mask] = block_id
            block_id += 1
    
    # Ensure we have the right number of blocks
    unique_blocks = np.unique(blocks)
    if len(unique_blocks) < n_blocks:
        print(f"  Note: Created {len(unique_blocks)} spatial blocks (requested {n_blocks})")
    
    return blocks

#==============================================================================
# BASELINE MODEL
#==============================================================================

def create_baseline_predictions(df, config):
    """Create rule-based baseline predictions"""
    print("\n" + "="*60)
    print("BASELINE MODEL (Rule-based)")
    print("="*60)
    
    # Check if required columns exist
    required_cols = ['NDVI_drop', 'NDWI_drop', 'h_AUC_z']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"Warning: Missing columns for baseline: {missing}")
        print("Skipping baseline model")
        return None, None
    
    # Apply Phase 4 rules
    primary = df['NDVI_drop'] > config.BASELINE_NDVI_DROP_THR
    auxiliary = (df['NDWI_drop'] > config.BASELINE_NDWI_DROP_THR) & \
                (df['h_AUC_z'] < config.BASELINE_AUC_Z_THR)
    
    # OR logic (as used in Phase 4)
    baseline_pred = (primary | auxiliary).astype(int)
    
    # Calculate metrics
    y_true = df[config.TARGET_COL].values
    
    accuracy = accuracy_score(y_true, baseline_pred)
    f1_stressed = f1_score(y_true, baseline_pred, pos_label=1)
    kappa = cohen_kappa_score(y_true, baseline_pred)
    
    print(f"Baseline Rules:")
    print(f"  NDVI_drop > {config.BASELINE_NDVI_DROP_THR}")
    print(f"  OR (NDWI_drop > {config.BASELINE_NDWI_DROP_THR} AND AUC_z < {config.BASELINE_AUC_Z_THR})")
    print(f"\nBaseline Performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1 (stressed): {f1_stressed:.3f}")
    print(f"  Cohen's κ: {kappa:.3f}")
    
    # Confusion matrix for baseline
    cm = confusion_matrix(y_true, baseline_pred)
    print(f"\nBaseline Confusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    return baseline_pred, {
        'accuracy': accuracy,
        'f1_stressed': f1_stressed,
        'kappa': kappa
    }

#==============================================================================
# RANDOM FOREST TRAINING
#==============================================================================

def train_random_forest(X, y, weights, spatial_blocks, config):
    """Train Random Forest with spatial cross-validation"""
    print("\n" + "="*60)
    print("RANDOM FOREST TRAINING")
    print("="*60)
    
    # Setup spatial cross-validation
    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    
    # Base estimator
    rf_base = RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS
    )
    
    # Random search with spatial CV
    print(f"\nRunning RandomizedSearchCV:")
    print(f"  Iterations: {config.N_ITER_SEARCH}")
    print(f"  CV folds: {config.CV_FOLDS} (spatial blocks)")
    print(f"  This may take several minutes...")
    
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=config.RF_PARAM_DISTRIBUTIONS,
        n_iter=config.N_ITER_SEARCH,
        cv=gkf.split(X, y, groups=spatial_blocks),
        scoring='f1',  # Optimize for stressed class F1
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        verbose=1
    )
    
    # Fit with sample weights if available
    if weights is not None:
        random_search.fit(X, y, sample_weight=weights)
    else:
        random_search.fit(X, y)
    
    print(f"\nBest parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV F1 score: {random_search.best_score_:.3f}")
    
    # Save hyperparameter search results
    search_results = pd.DataFrame(random_search.cv_results_)
    search_file = config.METRICS_DIR / 'hyperparameter_search_results.csv'
    search_results.to_csv(search_file, index=False)
    print(f"  Hyperparameter search results saved to: {search_file}")
    
    return random_search.best_estimator_

#==============================================================================
# MODEL EVALUATION
#==============================================================================

def evaluate_model(model, X, y, spatial_blocks, config):
    """Evaluate model with spatial cross-validation"""
    print("\n" + "="*60)
    print("MODEL EVALUATION (Spatial CV)")
    print("="*60)
    
    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    
    # Store predictions and metrics
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=spatial_blocks)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and train model for this fold
        fold_model = RandomForestClassifier(**model.get_params())
        fold_model.fit(X_train, y_train)
        
        # Predict
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1]
        
        # Store predictions
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        # Calculate fold metrics
        fold_metrics.append({
            'fold': fold + 1,
            'n_samples': len(y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_stressed': f1_score(y_test, y_pred, pos_label=1),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0
        })
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    # Overall metrics
    overall_metrics = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'f1_stressed': f1_score(all_y_true, all_y_pred, pos_label=1),
        'f1_healthy': f1_score(all_y_true, all_y_pred, pos_label=0),
        'kappa': cohen_kappa_score(all_y_true, all_y_pred),
        'roc_auc': roc_auc_score(all_y_true, all_y_proba)
    }
    
    # Print results
    print("\nPer-Fold Results:")
    print("-" * 60)
    fold_df = pd.DataFrame(fold_metrics)
    print(fold_df.to_string(index=False))
    
    print("\nFold Statistics:")
    print(f"  Mean Accuracy: {fold_df['accuracy'].mean():.3f} ± {fold_df['accuracy'].std():.3f}")
    print(f"  Mean F1 (stressed): {fold_df['f1_stressed'].mean():.3f} ± {fold_df['f1_stressed'].std():.3f}")
    print(f"  Mean κ: {fold_df['kappa'].mean():.3f} ± {fold_df['kappa'].std():.3f}")
    
    print("\nOverall Results (All Folds Combined):")
    print("-" * 60)
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\nClassification Report:")
    report = classification_report(all_y_true, all_y_pred, 
                                  target_names=['Healthy', 'Stressed'],
                                  digits=3)
    print(report)
    
    # Save classification report
    report_dict = classification_report(all_y_true, all_y_pred, 
                                       target_names=['Healthy', 'Stressed'],
                                       output_dict=True)
    report_file = config.METRICS_DIR / 'classification_report.json'
    with open(report_file, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    return overall_metrics, fold_metrics, all_y_true, all_y_pred, all_y_proba

#==============================================================================
# FEATURE IMPORTANCE AND SHAP
#==============================================================================

def analyze_feature_importance(model, X, feature_names, config):
    """Analyze feature importance using built-in and SHAP"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Built-in feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features (Gini):")
    print(importance.head(10).to_string(index=False))
    
    # SHAP analysis
    shap_values = None
    if SHAP_AVAILABLE:
        print("\nCalculating SHAP values...")
        try:
            # Use a subset for SHAP if dataset is large
            n_shap_samples = min(500, len(X))
            X_shap = X[:n_shap_samples]
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            
            # For binary classification, shap_values is a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use stressed class
            
            # Calculate mean absolute SHAP values
            shap_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)
            
            print("\nTop 10 Important Features (SHAP):")
            print(shap_importance.head(10).to_string(index=False))
            
            # Save SHAP importance
            shap_file = config.METRICS_DIR / 'shap_importance.csv'
            shap_importance.to_csv(shap_file, index=False)
            print(f"  SHAP importance saved to: {shap_file}")
            
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")
            shap_values = None
    
    return importance, shap_values

#==============================================================================
# VISUALIZATION
#==============================================================================

def create_visualizations(y_true, y_pred, y_proba, importance, shap_values, 
                         feature_names, baseline_metrics, rf_metrics, config):
    """Create all visualizations"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        # 1. Confusion Matrix and ROC Curve
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Healthy', 'Stressed'],
                    yticklabels=['Healthy', 'Stressed'])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Add percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(2):
            for j in range(2):
                axes[0].text(j + 0.5, i + 0.7, f'({cm_norm[i, j]:.1%})',
                            ha='center', va='center', fontsize=9, color='gray')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        axes[1].plot(fpr, tpr, label=f'RF (AUC = {rf_metrics["roc_auc"]:.3f})', linewidth=2)
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(config.FIGURE_DIR / 'confusion_roc.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance.head(15)
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance (Gini)')
        ax.set_title('Top 15 Feature Importance')
        ax.invert_yaxis()
        
        # Color bars by feature type
        colors = []
        for feat in top_features['feature']:
            if 'NDVI' in feat: colors.append('green')
            elif 'NDWI' in feat: colors.append('blue')
            elif 'GNDVI' in feat: colors.append('lightgreen')
            elif 'SAVI' in feat or 'MSAVI' in feat: colors.append('orange')
            elif 'AUC_z' in feat: colors.append('red')
            else: colors.append('gray')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig(config.FIGURE_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Model Comparison (if baseline exists)
        if baseline_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics_comparison = pd.DataFrame({
                'Baseline': [baseline_metrics['accuracy'], 
                            baseline_metrics['f1_stressed'], 
                            baseline_metrics['kappa']],
                'Random Forest': [rf_metrics['accuracy'], 
                                rf_metrics['f1_stressed'], 
                                rf_metrics['kappa']]
            }, index=['Accuracy', 'F1 (Stressed)', 'Cohen\'s κ'])
            
            x = np.arange(len(metrics_comparison.index))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, metrics_comparison['Baseline'], width, 
                          label='Baseline', color='lightcoral', alpha=0.8)
            bars2 = ax.bar(x + width/2, metrics_comparison['Random Forest'], width,
                          label='Random Forest', color='skyblue', alpha=0.8)
            
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_comparison.index)
            ax.legend()
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(config.FIGURE_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. SHAP Summary Plot
        if SHAP_AVAILABLE and shap_values is not None:
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, feature_names=feature_names, 
                                 show=False, max_display=15)
                plt.title('SHAP Feature Importance')
                plt.tight_layout()
                plt.savefig(config.FIGURE_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"Warning: Could not create SHAP plot: {e}")
        
        print(f"✓ Visualizations saved to {config.FIGURE_DIR}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        traceback.print_exc()

#==============================================================================
# MODEL EXPORT
#==============================================================================

def save_model_and_metrics(model, metrics, importance, config):
    """Save trained model and metrics"""
    print("\n" + "="*60)
    print("SAVING MODEL AND METRICS")
    print("="*60)
    
    # Save model as pickle
    model_path = config.MODEL_DIR / 'rf_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")
    
    # Save model parameters for GEE
    gee_params = {
        'model_type': 'RandomForestClassifier',
        'n_estimators': int(model.n_estimators),
        'max_depth': int(model.max_depth) if model.max_depth else None,
        'min_samples_split': int(model.min_samples_split),
        'min_samples_leaf': int(model.min_samples_leaf),
        'max_features': model.max_features,
        'feature_names': config.FEATURE_COLS,
        'feature_importances': model.feature_importances_.tolist(),
        'n_classes': int(model.n_classes_),
        'classes': model.classes_.tolist()
    }
    
    gee_path = config.MODEL_DIR / 'rf_model_gee.json'
    with open(gee_path, 'w') as f:
        json.dump(gee_params, f, indent=2)
    print(f"✓ GEE parameters saved to {gee_path}")
    
    # Save metrics
    metrics_path = config.METRICS_DIR / 'performance_metrics.json'
    
    # Convert any numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    metrics_clean = convert_numpy(metrics)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Save feature importance
    importance_path = config.METRICS_DIR / 'feature_importance.csv'
    importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to {importance_path}")

def save_final_summary(config, log_filename, baseline_metrics, rf_metrics):
    """Save a final summary report"""
    summary_file = config.OUTPUT_DIR / 'phase_6_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PHASE 6 SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("FINAL PERFORMANCE:\n")
        if baseline_metrics:
            f.write(f"- Baseline F1 (stressed): {baseline_metrics['f1_stressed']:.3f}\n")
            f.write(f"- RF F1 (stressed): {rf_metrics['f1_stressed']:.3f}\n")
            f.write(f"- Improvement: {rf_metrics['f1_stressed'] - baseline_metrics['f1_stressed']:+.3f}\n")
        else:
            f.write(f"- RF F1 (stressed): {rf_metrics['f1_stressed']:.3f}\n")
        
        f.write(f"- RF Accuracy: {rf_metrics['accuracy']:.3f}\n")
        f.write(f"- RF Cohen's kappa: {rf_metrics['kappa']:.3f}\n")
        f.write(f"- RF ROC-AUC: {rf_metrics['roc_auc']:.3f}\n")
        
        f.write(f"\nLog file saved to: {log_filename}\n")
        f.write("="*60 + "\n")
    
    print(f"\n📄 Summary report saved to: {summary_file}")

#==============================================================================
# MAIN EXECUTION
#==============================================================================

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Setup directories
    setup_directories(config)
    
    # Setup logging to file - FIXED FILENAME (overwrites previous)
    log_filename = config.LOG_DIR / "phase_6_log.txt"
    sys.stdout = Logger(log_filename)
    
    try:
        print("\n" + "="*60)
        print("PHASE 6: MODEL TRAINING & EVALUATION")
        print("="*60)
        
        # Validate inputs
        validate_inputs(config)
        
        # Load data
        X, y, weights, spatial_blocks, df = load_and_prepare_data(config)
        
        # Create baseline predictions
        baseline_pred, baseline_metrics = create_baseline_predictions(df, config)
        
        # Train Random Forest
        rf_model = train_random_forest(X, y, weights, spatial_blocks, config)
        
        # Evaluate model
        rf_metrics, fold_metrics, y_true, y_pred, y_proba = evaluate_model(
            rf_model, X, y, spatial_blocks, config
        )
        
        # Analyze feature importance
        importance, shap_values = analyze_feature_importance(
            rf_model, X, config.FEATURE_COLS, config
        )
        
        # Create visualizations
        create_visualizations(
            y_true, y_pred, y_proba, importance, shap_values,
            config.FEATURE_COLS, baseline_metrics, rf_metrics, config
        )
        
        # Save model and metrics
        all_metrics = {
            'baseline': baseline_metrics,
            'random_forest': rf_metrics,
            'fold_metrics': fold_metrics,
            'training_samples': len(X),
            'n_features': len(config.FEATURE_COLS),
            'feature_names': config.FEATURE_COLS
        }
        save_model_and_metrics(rf_model, all_metrics, importance, config)
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nFinal Performance Summary:")
        if baseline_metrics:
            print(f"  Baseline F1 (stressed): {baseline_metrics['f1_stressed']:.3f}")
            print(f"  RF F1 (stressed): {rf_metrics['f1_stressed']:.3f}")
            print(f"  Improvement: {rf_metrics['f1_stressed'] - baseline_metrics['f1_stressed']:+.3f}")
        else:
            print(f"  RF F1 (stressed): {rf_metrics['f1_stressed']:.3f}")
            print(f"  RF Accuracy: {rf_metrics['accuracy']:.3f}")
            print(f"  RF Cohen's κ: {rf_metrics['kappa']:.3f}")
        
        print(f"\n✓ All outputs saved to: {config.OUTPUT_DIR}")
        print(f"  - Models: {config.MODEL_DIR}")
        print(f"  - Figures: {config.FIGURE_DIR}")
        print(f"  - Metrics: {config.METRICS_DIR}")
        
        # Save final summary
        save_final_summary(config, log_filename, baseline_metrics, rf_metrics)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        traceback.print_exc()
    
    finally:
        print("\n" + "="*60)
        print("PHASE 6 COMPLETE")
        print("="*60)
        print(f"\n📝 Full log saved to: {log_filename}")
        
        # Close logger and restore stdout
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()
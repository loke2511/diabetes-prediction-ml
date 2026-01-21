"""
Diabetes Prediction - Model Training Script
Trains multiple ML models and saves the best performing one
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class DiabetesModelTrainer:
    """
    A comprehensive trainer for diabetes prediction models
    """
    
    def __init__(self, data_path='data/diabetes.csv'):
        """Initialize the trainer with data"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the diabetes dataset"""
        print("Loading data...")
        
        # Try to load from file, if not available, create sample data
        try:
            self.df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print("Dataset not found. Creating sample dataset...")
            self.df = self.create_sample_dataset()
            self.df.to_csv(self.data_path, index=False)
            print(f"Sample dataset saved to {self.data_path}")
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nDataset info:\n{self.df.info()}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nTarget distribution:\n{self.df['Outcome'].value_counts()}")
        
    def create_sample_dataset(self):
        """Create a sample Pima Indians Diabetes dataset"""
        # This is a simplified version - in production, use the real dataset
        np.random.seed(42)
        n_samples = 768
        
        # Generate sample data based on typical ranges for diabetes dataset
        data = {
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'Glucose': np.random.randint(0, 200, n_samples),
            'BloodPressure': np.random.randint(0, 122, n_samples),
            'SkinThickness': np.random.randint(0, 99, n_samples),
            'Insulin': np.random.randint(0, 846, n_samples),
            'BMI': np.random.uniform(0, 67.1, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
            'Age': np.random.randint(21, 81, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create outcome based on some logical rules (simplified)
        outcome = np.zeros(n_samples)
        for i in range(n_samples):
            score = 0
            if df.loc[i, 'Glucose'] > 140: score += 2
            if df.loc[i, 'BMI'] > 30: score += 1
            if df.loc[i, 'Age'] > 45: score += 1
            if df.loc[i, 'Insulin'] > 200: score += 1
            if df.loc[i, 'BloodPressure'] > 80: score += 1
            
            # Add some randomness
            if score >= 3 or (score >= 2 and np.random.random() > 0.5):
                outcome[i] = 1
        
        df['Outcome'] = outcome.astype(int)
        return df
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        # Handle zero values (they might represent missing data in this dataset)
        # Replace zeros with median for certain columns
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in X.columns:
                X[col] = X[col].replace(0, X[col].median())
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def initialize_models(self):
        """Initialize all models to be trained"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
        }
        
        print(f"Initialized {len(self.models)} models")
        
    def train_and_evaluate(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        print("=" * 80)
        
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='accuracy')
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "=" * 80)
        print(f"\nBest Model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
    def save_models(self):
        """Save the best model and scaler"""
        print("\nSaving models...")
        
        # Save best model
        joblib.dump(self.best_model, 'models/best_model.pkl')
        print(f"Best model ({self.best_model_name}) saved to models/best_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Scaler saved to models/scaler.pkl")
        
        # Save all models
        for name, model in self.models.items():
            safe_name = name.replace(' ', '_').lower()
            joblib.dump(model, f'models/{safe_name}.pkl')
        print(f"All {len(self.models)} models saved")
        
        # Save results
        results_dict = {
            'best_model': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': self.results
        }
        
        with open('models/training_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        print("Training results saved to models/training_results.json")
        
    def create_visualizations(self):
        """Create visualizations of the results"""
        print("\nCreating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Diabetes Prediction - Model Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        model_names = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        precisions = [self.results[m]['precision'] for m in model_names]
        recalls = [self.results[m]['recall'] for m in model_names]
        f1_scores = [self.results[m]['f1_score'] for m in model_names]
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        colors = sns.color_palette("husl", len(model_names))
        bars1 = ax1.barh(model_names, accuracies, color=colors)
        ax1.set_xlabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_xlim([0, 1])
        
        # Add value labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{accuracies[i]:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. Precision, Recall, F1 comparison
        ax2 = axes[0, 1]
        x = np.arange(len(model_names))
        width = 0.25
        
        ax2.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax2.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_title('Precision, Recall, and F1-Score Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim([0, 1])
        
        # 3. Confusion Matrix for best model
        ax3 = axes[1, 0]
        cm = np.array(self.results[self.best_model_name]['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
                   cbar_kws={'label': 'Count'})
        ax3.set_title(f'Confusion Matrix - {self.best_model_name}', fontweight='bold')
        ax3.set_ylabel('True Label', fontweight='bold')
        ax3.set_xlabel('Predicted Label', fontweight='bold')
        ax3.set_xticklabels(['No Diabetes', 'Diabetes'])
        ax3.set_yticklabels(['No Diabetes', 'Diabetes'])
        
        # 4. ROC-AUC scores
        ax4 = axes[1, 1]
        roc_aucs = [self.results[m]['roc_auc'] for m in model_names]
        bars4 = ax4.barh(model_names, roc_aucs, color=colors)
        ax4.set_xlabel('ROC-AUC Score', fontweight='bold')
        ax4.set_title('ROC-AUC Score Comparison', fontweight='bold')
        ax4.set_xlim([0, 1])
        
        # Add value labels
        for i, bar in enumerate(bars4):
            width = bar.get_width()
            ax4.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{roc_aucs[i]:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('static/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to static/model_comparison.png")
        plt.close()
        
        # Create feature importance plot for Random Forest
        if 'Random Forest' in self.models:
            self.plot_feature_importance()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        print("Creating feature importance plot...")
        
        plt.figure(figsize=(10, 6))
        
        # Get feature importance from Random Forest
        rf_model = self.models['Random Forest']
        feature_names = self.df.drop('Outcome', axis=1).columns
        importances = rf_model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.bar(range(len(importances)), importances[indices], color='teal', alpha=0.8)
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features', fontweight='bold', fontsize=12)
        plt.ylabel('Importance', fontweight='bold', fontsize=12)
        plt.title('Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to static/feature_importance.png")
        plt.close()
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "=" * 80)
        print("DIABETES PREDICTION MODEL TRAINING PIPELINE")
        print("=" * 80)
        
        self.load_data()
        self.preprocess_data()
        self.initialize_models()
        self.train_and_evaluate()
        self.save_models()
        self.create_visualizations()
        
        print("\n" + "=" * 80)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print("\nFiles created:")
        print("  - models/best_model.pkl")
        print("  - models/scaler.pkl")
        print("  - models/training_results.json")
        print("  - static/model_comparison.png")
        print("  - static/feature_importance.png")
        print("\nYou can now run the Flask app with: python app.py")


def main():
    """Main function"""
    trainer = DiabetesModelTrainer()
    trainer.run_full_pipeline()


if __name__ == '__main__':
    main()

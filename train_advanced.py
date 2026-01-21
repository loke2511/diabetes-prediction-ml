"""
Advanced Model Training with SHAP Explanations and Calibration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
import warnings
warnings.filterwarnings('ignore')

class AdvancedDiabetesTrainer:
    """Advanced trainer with SHAP and calibration analysis"""
    
    def __init__(self, data_path='data/diabetes.csv'):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load the PIMA dataset"""
        print("Loading PIMA Indians Diabetes Dataset...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✓ Real PIMA dataset loaded!")
        except FileNotFoundError:
            print("❌ Dataset not found. Run download_dataset.py first!")
            raise
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nDataset info:\n{self.df.info()}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nTarget distribution:\n{self.df['Outcome'].value_counts()}")
        
    def preprocess_data(self):
        """Advanced data preprocessing"""
        print("\nPreprocessing data...")
        
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        # Handle zero values (missing data indicators in PIMA dataset)
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_cols:
            if col in X.columns:
                # Replace zeros with NaN, then impute with median
                X[col] = X[col].replace(0, np.nan)
                X[col].fillna(X[col].median(), inplace=True)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Features: {', '.join(self.feature_names)}")
        
    def initialize_models(self):
        """Initialize all models"""
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
        
    def train_and_evaluate(self):
        """Train and evaluate all models"""
        print("\nTraining models...")
        print("=" * 80)
        
        best_accuracy = 0
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            model.fit(self.X_train_scaled, self.y_train)
            
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='accuracy')
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
            
            print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                  f"Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print("\n" + "=" * 80)
        print(f"🏆 Best Model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
    def create_shap_explanations(self):
        """Generate SHAP values for model interpretability"""
        print("\n🔍 Generating SHAP explanations...")
        
        try:
            # Use Random Forest for SHAP (tree-based models work best)
            rf_model = self.models['Random Forest']
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(self.X_test_scaled)
            
            # If binary classification, shap_values might be a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.feature_names,
                            show=False)
            plt.tight_layout()
            plt.savefig('static/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ SHAP summary plot saved")
            
            # Feature importance from SHAP
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_test, 
                            feature_names=self.feature_names,
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('static/shap_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ SHAP importance plot saved")
            
            # Save SHAP values for later use
            np.save('models/shap_values.npy', shap_values)
            
        except Exception as e:
            print(f"⚠ SHAP generation failed: {e}")
    
    def create_calibration_plots(self):
        """Create calibration plots for probability predictions"""
        print("\n📊 Creating calibration plots...")
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, (name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            # Get predicted probabilities
            y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.y_test, y_prob, n_bins=10
            )
            
            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', linewidth=2, label=name)
            
            # Plot perfect calibration
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            
            ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontweight='bold')
            ax.set_title(f'{name}\nCalibration Curve', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide the last subplot if there are only 7 models
        if len(self.models) < 8:
            axes[-1].axis('off')
        
        plt.suptitle('Model Calibration Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('static/calibration_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Calibration plots saved")
    
    def create_roc_curves(self):
        """Create ROC curves for all models"""
        print("\n📈 Creating ROC curves...")
        
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            y_prob = model.predict_proba(self.X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        plt.title('ROC Curves - All Models', fontweight='bold', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ ROC curves saved")
    
    def save_models(self):
        """Save all models and results"""
        print("\n💾 Saving models...")
        
        joblib.dump(self.best_model, 'models/best_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        for name, model in self.models.items():
            safe_name = name.replace(' ', '_').lower()
            joblib.dump(model, f'models/{safe_name}.pkl')
        
        # Save results
        results_dict = {
            'best_model': self.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': 'Real PIMA Indians Diabetes Dataset',
            'models': self.results,
            'feature_names': self.feature_names
        }
        
        with open('models/training_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print("✓ All models and results saved")
    
    def run_full_pipeline(self):
        """Run complete advanced training pipeline"""
        print("\n" + "=" * 80)
        print("🚀 ADVANCED DIABETES PREDICTION TRAINING PIPELINE")
        print("=" * 80)
        
        self.load_data()
        self.preprocess_data()
        self.initialize_models()
        self.train_and_evaluate()
        self.create_shap_explanations()
        self.create_calibration_plots()
        self.create_roc_curves()
        self.save_models()
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"📊 Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"🎯 ROC-AUC: {self.results[self.best_model_name]['roc_auc']:.4f}")
        print("\n📁 Files created:")
        print("  ✓ models/best_model.pkl")
        print("  ✓ models/scaler.pkl")
        print("  ✓ models/training_results.json")
        print("  ✓ static/shap_summary.png (NEW!)")
        print("  ✓ static/shap_importance.png (NEW!)")
        print("  ✓ static/calibration_plots.png (NEW!)")
        print("  ✓ static/roc_curves.png (NEW!)")

def main():
    trainer = AdvancedDiabetesTrainer()
    trainer.run_full_pipeline()

if __name__ == '__main__':
    main()

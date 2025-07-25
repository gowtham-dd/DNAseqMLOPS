from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.DNASeqMLOPS.entity.config_entity import ModelTrainerConfig
import os
from typing import Dict
from DNASeqMLOPS import logger
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model_classes = {
            "RandomForest": RandomForestClassifier,
            "SVM": SVC,
            "XGBoost": xgb.XGBClassifier
        }
        
    def train(self):
        """Train models only if no trained models exist yet"""
        model_files_exist = any(
            fname.endswith(".joblib") for fname in os.listdir(self.config.model_dir)
        ) if os.path.exists(self.config.model_dir) else False

        if model_files_exist:
            logger.info(f"Model files already exist in {self.config.model_dir} - skipping training")
            return False
        else:
            os.makedirs(self.config.model_dir, exist_ok=True)
            logger.info("Training models...")

            X = np.load(os.path.join(self.config.transformed_data_dir, "X.npy"))
            y = np.load(os.path.join(self.config.transformed_data_dir, "y.npy"))

            results = {}
            for model_name, params in self.config.models_params.items():
                if model_name in self.model_classes:
                    try:
                        logger.info(f"Training {model_name} with params: {params}")
                        model = self.model_classes[model_name](**params)
                        model.fit(X, y)

                    # Save model
                        model_path = os.path.join(self.config.model_dir, f"{model_name}.joblib")
                        joblib.dump(model, model_path)

                    # Evaluate
                        y_pred = model.predict(X)
                        results[model_name] = {
                        "accuracy": accuracy_score(y, y_pred),
                        "f1": f1_score(y, y_pred),
                        "classification_report": classification_report(y, y_pred)
                        }
                        logger.info(f"{model_name} trained successfully")

                    except Exception as e:
                        logger.error(f"Error training {model_name}: {str(e)}")
                        continue

        # Save evaluation results
            joblib.dump(results, os.path.join(self.config.model_dir, "training_results.joblib"))
            logger.info("Model training completed")
            return True

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
import json
import mlflow
import os
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from DNASeqMLOPS import logger
from DNASeqMLOPS.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        # Create evaluation directory if it doesn't exist
        os.makedirs(self.config.root_dir, exist_ok=True)

    def _load_test_data(self):
        """Load test features and labels"""
        try:
            X_test = np.load(self.config.test_data_path)
            y_test = np.load(self.config.test_labels_path)
            return X_test, y_test
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise

    def _load_models(self):
        """Load all trained models"""
        models = {}
        try:
            for model_file in os.listdir(self.config.model_dir):
                if model_file.endswith('.joblib'):
                    model_name = model_file.split('.')[0]
                    model_path = os.path.join(self.config.model_dir, model_file)
                    models[model_name] = joblib.load(model_path)
            return models
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def evaluate_models(self):
        """Evaluate all models and return metrics"""
        try:
            X_test, y_test = self._load_test_data()
            models = self._load_models()
            
            metrics = {}
            for model_name, model in models.items():
                try:
                    y_pred = model.predict(X_test)
                    
                    metrics[model_name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                    }
                    logger.info(f"\n{model_name} Evaluation:\n{json.dumps(metrics[model_name], indent=2)}")
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    continue
            
            return metrics
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _save_metrics(self, metrics):
        """Save metrics to JSON file"""
        try:
            with open(self.config.metric_file_name, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.config.metric_file_name}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise

    def log_into_mlflow(self):
        """Log evaluation results to MLflow"""
        try:
            if not os.path.exists(self.config.metric_file_name):
                mlflow.set_tracking_uri(self.config.mlflow_uri)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                with mlflow.start_run():
                    # Evaluate models
                    metrics = self.evaluate_models()
                    
                    # Save metrics to file
                    self._save_metrics(metrics)
                    
                    # Log parameters and metrics
                    mlflow.log_params(self.config.all_params)
                    for model_name, model_metrics in metrics.items():
                        for metric_name, value in model_metrics.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"{model_name}_{metric_name}", value)
                    
                    # Log models
                    if tracking_url_type_store != "file":
                        models = self._load_models()
                        for model_name, model in models.items():
                            try:
                                if hasattr(model, 'predict'):
                                    mlflow.sklearn.log_model(
                                        sk_model=model,
                                        artifact_path=f"{model_name}_model",
                                        registered_model_name=f"DNA_Seq_{model_name}"
                                    )
                            except Exception as e:
                                logger.error(f"Error logging {model_name} to MLflow: {e}")
                    
                    logger.info("Evaluation results logged to MLflow")
                return True
            else:
                logger.info(f"Metrics file {self.config.metric_file_name} already exists - skipping evaluation")
                return False
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            raise
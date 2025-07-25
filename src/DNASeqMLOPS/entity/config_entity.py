from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_path: Path 
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path  # Input data (trainYX.csv)
    transformed_features_path: Path  # Output features (X.npy)
    transformed_labels_path: Path  # Output labels (y.npy)



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    transformed_data_dir: Path
    model_dir: Path
    models_params: dict  # Will contain params from params.yaml



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    test_labels_path: Path
    model_dir: Path
    metric_file_name: Path
    mlflow_uri: str
    all_params: dict
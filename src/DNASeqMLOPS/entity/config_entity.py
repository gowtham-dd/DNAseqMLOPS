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

from src.DNASeqMLOPS.constant import *
from src.DNASeqMLOPS.utils.common import read_yaml,create_directories 
from src.DNASeqMLOPS.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    


    def get_data_validation_config(self) -> DataValidationConfig:
        """Get the data validation configuration for single dataset"""
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            data_path=Path(config.data_path),  # Single data file
            all_schema=schema
        )
   

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            transformed_features_path=Path(config.transformed_features_path),
            transformed_labels_path=Path(config.transformed_labels_path)
        )
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        models_params = self.params.model_training.models
        
        create_directories([
            Path(config.root_dir),
            Path(config.model_dir)
        ])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            transformed_data_dir=Path(config.transformed_data_dir),
            model_dir=Path(config.model_dir),
            models_params=models_params
        )
    
        
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            test_labels_path=Path(config.test_labels_path),
            model_dir=Path(config.model_dir),
            metric_file_name=Path(config.metric_file_name),
            mlflow_uri=config.mlflow_uri,
            all_params=self.params.model_training  # All model params from params.yaml
        )
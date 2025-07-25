from src.DNASeqMLOPS.config.configuration import ConfigurationManager
from src.DNASeqMLOPS.components.Data_Validation import DataValidation
from src.DNASeqMLOPS import logger  
import pandas as pd

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
    # Initialize configuration manager
            config_manager = ConfigurationManager()
    
    # Get data validation config
            data_validation_config = config_manager.get_data_validation_config()
    
    # Perform validation
            validator = DataValidation(config=data_validation_config)
            is_valid = validator.validate_dataset()
    
            if not is_valid:
                raise ValueError("Data validation failed - check logs for details")
    
            print("Data validation passed successfully")
    
    # Now you can proceed to split the validated data into train/test sets
            df = pd.read_csv(data_validation_config.data_path)
    # ... (split logic here)
    
        except Exception as e:
            print(f"Error during data validation: {str(e)}")
            raise e

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

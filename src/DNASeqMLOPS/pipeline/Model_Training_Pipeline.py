

from src.DNASeqMLOPS.config.configuration import ConfigurationManager
from src.DNASeqMLOPS.components.Model_Training import ModelTrainer
from src.DNASeqMLOPS import logger  
import pandas as pd

STAGE_NAME = "Data Validation stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            trainer_config = config_manager.get_model_trainer_config()
            trainer = ModelTrainer(trainer_config)
    
            if trainer.train():
                logger.info("New models trained and saved")
            else:
                logger.info("Using existing models")
        
        except Exception as e:
            logger.error(f"Model training pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

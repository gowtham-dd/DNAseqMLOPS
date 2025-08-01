from src.DNASeqMLOPS.config.configuration import ConfigurationManager
from src.DNASeqMLOPS.components.Data_Ingestion import DataIngestion
from src.DNASeqMLOPS import logger


STAGE_NAME="Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
           config = ConfigurationManager()
           data_ingestion_config = config.get_data_ingestion_config()
           data_ingestion = DataIngestion(config=data_ingestion_config)
           data_ingestion.download_file()
           data_ingestion.extract_zip_file()


if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataIngestionTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e

from src.DNASeqMLOPS.pipeline.Data_Validation_Pipeline import DataValidationTrainingPipeline 
from src.DNASeqMLOPS.pipeline.Data_Transformation_Pipeline import DataTransformationPipeline
from src.DNASeqMLOPS.pipeline.Model_Training_Pipeline import ModelTrainingPipeline
from src.DNASeqMLOPS.pipeline.Model_Evaluation_Pipelnie import ModelEvaluationTrainingPipeline
# import dagshub
# dagshub.init(repo_owner='gowtham-dd', repo_name='winepred-MLFLOW', mlflow=True)
from src.DNASeqMLOPS.pipeline.Data_INgestion_Pipeline import DataIngestionTrainingPipeline
from src.DNASeqMLOPS import logger

STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Data Validation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Data Transformation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e





STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e

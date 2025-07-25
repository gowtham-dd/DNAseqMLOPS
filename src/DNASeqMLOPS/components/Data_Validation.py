from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_path: Path  # Single data file that will be split later
    all_schema: Dict

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self) -> bool:
        """
        Validate that the dataset contains the required columns
        with the correct data types as specified in the schema.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            validation_status = True
            expected_columns = set(self.config.all_schema.keys())

            # Read the dataset
            df = pd.read_csv(self.config.data_path)
            
            # Check all required columns are present
            actual_columns = set(df.columns)
            if not expected_columns.issubset(actual_columns):
                missing_cols = expected_columns - actual_columns
                print(f"Missing columns: {missing_cols}")
                validation_status = False
            
            # Check data types for each column
            for col, props in self.config.all_schema.items():
                if col not in df.columns:
                    continue
                
                # Check data type
                expected_type = props['type']
                actual_type = str(df[col].dtype)
                
                # Handle type variations
                if expected_type == 'int' and 'int' in actual_type:
                    continue
                if expected_type == 'float' and 'float' in actual_type:
                    continue
                if expected_type == 'string' and actual_type == 'object':  # Pandas stores strings as object
                    continue
                if expected_type != actual_type:
                    print(f"Type mismatch in column '{col}': "
                          f"expected {expected_type}, got {actual_type}")
                    validation_status = False
            
            # Additional DNA-specific validations
            if 'DNA' in df.columns:
                # Check DNA sequences only contain valid nucleotides
                valid_nucleotides = {'A', 'T', 'C', 'G'}
                sample_sequences = df['DNA'].sample(min(100, len(df)))
                for seq in sample_sequences:
                    if not set(seq).issubset(valid_nucleotides):
                        print(f"Invalid nucleotides found in DNA sequence: {seq}")
                        validation_status = False
                        break
            
            # Write validation status to file
            with open(Path(self.config.root_dir) , 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            return validation_status
            
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            raise e
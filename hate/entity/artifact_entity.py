from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str

@dataclass
class ModelTrainerArtifacts: 
    trained_model_path:str
    x_test_file_path: str
    y_test_file_path: str

# Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool    
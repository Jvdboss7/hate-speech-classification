import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.ml.model import ModelArchitecture
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts,DataTransformationArtifacts


class ModelTrainer:
    def __init__(self,data_transformation_artifacts: DataTransformationArtifacts,
                model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
        

    def spliting_data(self,csv_path):
        try:
            logging.info("Reading the data")
            df = pd.read_csv(csv_path)
            logging.info("Splitting the data into x and y")
            x = df[self.model_trainer_config.TWEET]
            y = df[self.model_trainer_config.LABEL]

            logging.info("Applying train_test_split on the data")
            x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 42)
            print(len(x_train),len(y_train))
            print(len(x_test),len(y_test))

            return x_train,x_test,y_train,y_test

        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self,x_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
            return sequences_matrix,tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            x_train,x_test,y_train,y_test = self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)

            sequences_matrix,tokenizer =self.tokenizing(x_train)

            model_architecture = ModelArchitecture(model_trainer_config = self.model_trainer_config)

            model = model_architecture.get_model()
        
            model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        # callbacks=[model_architecture.early_stopping]
                        )

            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)

            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH
            )
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

# Creating model architecture.
from hate.entity.config_entity import ModelTrainerConfig
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from hate.constants import *
class ModelArchitecture:

    def __init__(self,model_trainer_config:ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def early_stopping(self):
        # Early stopping and callbacks
        stop =  EarlyStopping(
            monitor = 'val_accuracy',
            mode = 'max',
            patience = 5)
    def callbacks(self):
        checkpoint = ModelCheckpoint(
            filepath='./',
            save_weights_only = True,
            monitor = 'val_accuracy',
            mode = 'max',
            save_best_only = True)

    def get_model(self):
        model = Sequential()
        model.add(Embedding(self.model_trainer_config.MAX_WORDS, 100,input_length=self.model_trainer_config.MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1,activation=self.model_trainer_config.ACTIVATION))
        model.summary()
        model.compile(loss=self.model_trainer_config.LOSS,optimizer=RMSprop(),metrics=self.model_trainer_config.METRICS)

        return model
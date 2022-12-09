from hate.pipeline.train_pipeline import TrainPipeline
from fastapi import FastAPI
import uvicorn
import sys
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from hate.pipeline.prediction_pipeline import PredictionPipeline
# from text_generation.ml.model.prediction import TextGenerator
from hate.exception import CustomException


text:str = "What is machine learing?"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text):
    try:
        # logging.info("Downloading model from s3 bucket")
        # os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        # s3_sync = s3sync()
        # aws_buket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
        # s3_sync.sync_folder_from_s3(folder = SAVED_MODEL_DIR,aws_bucket_url=aws_buket_url)
        # logging.info("Model downloading completed")
        # obj= TextGenerator()
        # generated_text = obj.prediction(text)
        obj = PredictionPipeline()
        text = obj.run_pipeline(text)
        return text
    except Exception as e:
        raise CustomException(e, sys) from e

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
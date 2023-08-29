import fastapi
import pandas as pd
from typing import List, Dict
from model import DelayModel

app = fastapi.FastAPI()
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data_point: List[Dict[str, str]]) -> dict:
    # Preprocess the input data
    input_data = pd.DataFrame(data_point)
    preprocessed_data = model.preprocess(input_data)
    
    # Predict using the trained model
    predictions = model.predict(preprocessed_data)
    
    return {"predictions": predictions}

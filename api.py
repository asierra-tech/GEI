from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained models
preprocessor_habitaciones_banios = load('property_preprocessor.habitaciones_banios.joblib')
pca_habitaciones_banios = load('property_pca.habitaciones_banios.joblib')

preprocessor_caracteristicas = load('property_preprocessor.caracteristicas.joblib')
pca_caracteristicas = load('property_pca.caracteristicas.joblib')

# Define the input schema using Pydantic
class PropertyDetails(BaseModel):
    habitaciones: int
    banios: int   
    trastero: bool
    garaje: bool
    piscina: bool
    ac: bool
    ascensor: bool
    terraza: bool
    jardin: bool

# Helper functions to calculate embeddings
def calculate_embedding_habitaciones_banios(property_df):
    input_df = property_df[['habitaciones', 'banios']]
    processed_data = preprocessor_habitaciones_banios.transform(input_df)
    embedding = pca_habitaciones_banios.transform(processed_data)
    if embedding.shape[1] < 64:
        padded_embedding = np.zeros((1, 64))
        padded_embedding[0, :embedding.shape[1]] = embedding
        return padded_embedding[0].tolist()
    return embedding[0].tolist()

def calculate_embedding_caracteristicas(property_df):
    boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
    for col in boolean_features:
        property_df[col] = property_df[col].astype(int)
    input_df = property_df[boolean_features]
    processed_data = preprocessor_caracteristicas.transform(input_df)
    embedding = pca_caracteristicas.transform(processed_data)
    if embedding.shape[1] < 64:
        padded_embedding = np.zeros((1, 64))
        padded_embedding[0, :embedding.shape[1]] = embedding
        return padded_embedding[0].tolist()
    return embedding[0].tolist()

# Define the POST endpoint
@app.post("/calculate-embeddings/")
async def calculate_embeddings(property_details: PropertyDetails):
    try:
        # Convert input to DataFrame
        property_df = pd.DataFrame([property_details.dict()])

        # Calculate embeddings
        embedding_habitaciones_banios = calculate_embedding_habitaciones_banios(property_df)
        embedding_caracteristicas = calculate_embedding_caracteristicas(property_df)

        # Return embeddings as JSON
        return {
            "embedding_habitaciones_banios": embedding_habitaciones_banios,
            "embedding_caracteristicas": embedding_caracteristicas
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
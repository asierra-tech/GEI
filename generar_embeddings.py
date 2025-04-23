import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import psycopg2
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine
from dotenv import load_dotenv
from psycopg2.extras import execute_batch
from joblib import dump, load

CHUNK_SIZE=2000000 # Number of records to process at a time

# Load environment variables
load_dotenv()

def get_db_connection_params():
    """Get database connection parameters from environment"""
    # params = {
    #     "DB_NAME": "betipo-valoracion-dev",
    #     "DB_USER": "doadmin",
    #     "DB_PASSWORD": "AVNS_z5jgGVGmqBsRCmMukgc",
    #     "DB_HOST": "pg-betipo-do-user-20048063-0.g.db.ondigitalocean.com",
    #     "DB_PORT": "25060"
    # }
    params = {
        "DB_NAME": os.getenv("DB_NAME"),
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT")
    }

    if None in params.values():
        raise ValueError("Missing environment variables for DB connection")
    
    return params

def fetch_property_data(engine, chunk_size=10000):
    """Fetch property data from database in chunks"""
    query = """
    SELECT 
        id, m2, habitaciones, banios, planta, 
        tipo_inmueble, estado_conservacion, 
        trastero, garaje, piscina, ac, ascensor, terraza, jardin
    FROM testigos
    """
    for chunk in pd.read_sql(query, engine, chunksize=chunk_size):
        yield chunk

def create_embedding_pipeline(numeric_features, categorical_features, boolean_features):
    """Create preprocessing pipeline for embeddings"""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Los null se reemplazan con el valor medio
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Los null se reemplazan con el valor más frecuente
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    boolean_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=False)), # Los null se reemplazan con False
        ('scaler', StandardScaler())
    ])
    
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('bool', boolean_transformer, boolean_features)
    ])

def generate_embeddings(data, preprocessor, embedding_dim=64):
    """Generate embeddings using PCA"""
    processed_data = preprocessor.fit_transform(data)
    
    n_components = min(embedding_dim, processed_data.shape[0], processed_data.shape[1])
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(processed_data)
    
    # Pad if necessary
    if n_components < embedding_dim:
        padded_embeddings = np.zeros((embeddings.shape[0], embedding_dim))
        padded_embeddings[:, :n_components] = embeddings
        return padded_embeddings
    
    return embeddings

def store_embeddings_batch(conn, df, embeddings, column_name, batch_size=1000):
    """Store embeddings in database using batch operations"""
    cursor = conn.cursor()
    
    # Prepare data for batch update
    data = [(np.array(emb), id_val) for emb, id_val in zip(embeddings, df['id'])]
    
    query = f"UPDATE testigos SET {column_name} = %s WHERE id = %s"
    
    # Execute in batches
    execute_batch(cursor, query, data, batch_size)
    
    conn.commit()
    cursor.close()

def calculate_embedding(conn, df, numeric_features, categorical_features, boolean_features, column_name):
    # Print info about missing values
    print("\nMissing values in features:")
    for col in numeric_features + categorical_features + boolean_features:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"{col}: {missing} missing values ({(missing/len(df))*100:.2f}%)")
    
    # Create preprocessing pipeline
    preprocessor = create_embedding_pipeline(numeric_features, categorical_features, boolean_features)
    
    # Convert boolean columns to proper boolean type
    for col in boolean_features:
        df[col] = df[col].astype('boolean')
    
    # Select only the columns we need
    features_df = df[numeric_features + categorical_features + boolean_features]
    
    # Generate embeddings
    embedding_dim = 64
    embeddings = generate_embeddings(features_df, preprocessor, embedding_dim)

    # Connect to database for vector operations
    register_vector(conn)
    
    # Store embeddings in batches
    store_embeddings_batch(conn, df, embeddings, column_name)
    
def calculate_vector_embedding(df, numeric_features=None, categorical_features=None, boolean_features=None, embedding_dim=64):
      # Print info about missing values
    print("\nMissing values in features:")
    for col in numeric_features + categorical_features + boolean_features:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"{col}: {missing} missing values ({(missing/len(df))*100:.2f}%)")
    
    # Create preprocessing pipeline
    preprocessor = create_embedding_pipeline(numeric_features, categorical_features, boolean_features)
    
    # Convert boolean columns to proper boolean type
    for col in boolean_features:
        df[col] = df[col].astype('boolean')
    
    # Select only the columns we need
    features_df = df[numeric_features + categorical_features + boolean_features]
    
    # Generate embeddings
    embedding_dim = 64
    embeddings = generate_embeddings(features_df, preprocessor, embedding_dim)
    return embeddings
     # Return single embedding vector

def store_pca_model(chunk, embeding_name,numeric_features, categorical_features, boolean_features):
    for col in boolean_features:
        chunk[col]=chunk[col].astype(int)
            # Create and fit global preprocessor
    global_preprocessor = create_embedding_pipeline(numeric_features, categorical_features, boolean_features)
            
            # Fit on first chunk
    features_df = chunk[numeric_features + categorical_features + boolean_features]
    processed_data = global_preprocessor.fit_transform(features_df)
            
            # Create and fit global PCA
    n_components = min(64, processed_data.shape[0], processed_data.shape[1])  # Adjust n_components dynamically
    global_pca = PCA(n_components=n_components)
    global_pca.fit(processed_data)
            
            # Save models
    dump(global_preprocessor, f"property_preprocessor.{embeding_name}.joblib")
    dump(global_pca, f"property_pca.{embeding_name}.joblib")

def main():
    # Get connection parameters
    params = get_db_connection_params()
    
    # Create SQLAlchemy engine
    engine = create_engine(
        f"postgresql+psycopg2://{params['DB_USER']}:{params['DB_PASSWORD']}@"
        f"{params['DB_HOST']}:{params['DB_PORT']}/{params['DB_NAME']}"
    )
    
    # Connect to database for vector operations
    conn = psycopg2.connect(
        dbname=params['DB_NAME'],
        user=params['DB_USER'],
        password=params['DB_PASSWORD'],
        host=params['DB_HOST'],
        port=params['DB_PORT']
    )

    global_preprocessor = None
    global_pca = None
    
    total_records = 0
    first_chunk = True

    # Process data in chunks
    for chunk in fetch_property_data(engine, chunk_size=CHUNK_SIZE):
        if first_chunk:
            # Define features for full embedding
            numeric_features = ['habitaciones', 'banios']
            categorical_features = []
            boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
            
            store_pca_model(chunk, "habitaciones_banios",numeric_features, [],[])
            store_pca_model(chunk, "caracteristicas",[], [],boolean_features)
            
            first_chunk = False
       
       ###############
        total_records += len(chunk)
        print(f"\nProcessing chunk of {len(chunk)} records. Total processed: {total_records}")
      
       # Define features
        numeric_features=[]
        categorical_features=[]
        #numeric_features = ['m2', 'habitaciones', 'banios', 'planta']
        #categorical_features = ['tipo_inmueble', 'estado_conservacion']
        boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
        
        calculate_embedding(conn, chunk, numeric_features, categorical_features, 
                          boolean_features, column_name='embedding')
    
     # Define features
        numeric_features = ['habitaciones', 'banios']
        categorical_features = []
        boolean_features = []
    
        calculate_embedding(conn, chunk, numeric_features, categorical_features, 
                          boolean_features, column_name='embedding_habitaciones_banios')
    
    conn.close()
    print(f"\nTotal records processed: {total_records}")
    print("Embeddings generated and stored successfully!")



def calculate_new_property_embedding_habitaciones_banios(new_property_df):
    """Calculate embedding for a single new property"""
    # Load saved models
    preprocessor = load('property_preprocessor.habitaciones_banios.joblib')
    pca = load('property_pca.habitaciones_banios.joblib')
    
    # Define features (must match training features)
    numeric_features = [ 'habitaciones', 'banios']
    categorical_features = []
    boolean_features = [] #['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
    
    # for col in boolean_features:
    #     new_property_df[col]=chunk[col].astype(int)
    # Prepare input data
    input_df = new_property_df[numeric_features + categorical_features + boolean_features]
    
    # Transform using pre-trained pipeline
    processed_data = preprocessor.transform(input_df)
    
    # Apply PCA
    embedding = pca.transform(processed_data)
    
    # Pad if necessary
    if embedding.shape[1] < 64:
        padded_embedding = np.zeros((1, 64))
        padded_embedding[0, :embedding.shape[1]] = embedding
        return padded_embedding
    
    return embedding

def calculate_new_property_embedding_caracteristicas(new_property_df):
    """Calculate embedding for a single new property"""
    # Load saved models
    preprocessor = load('property_pca.caracteristicas.joblib')
    pca = load('property_pca.caracteristicas.joblib')
    
    # Define features (must match training features)
    numeric_features = []
    categorical_features = []
    boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
    
        # Convert boolean features to integers
    for col in boolean_features:
        new_property_df[col] = new_property_df[col].astype(int)

    # for col in boolean_features:
    #     new_property_df[col]=chunk[col].astype(int)
    # Prepare input data
    input_df = new_property_df[numeric_features + categorical_features + boolean_features]
    
    # Transform using pre-trained pipeline
    processed_data = preprocessor.transform(input_df)
    
    # Apply PCA
    embedding = pca.transform(processed_data)
    
    # Pad if necessary
    if embedding.shape[1] < 64:
        padded_embedding = np.zeros((1, 64))
        padded_embedding[0, :embedding.shape[1]] = embedding
        return padded_embedding
    
    return embedding

def main1():
    new_property = pd.DataFrame([{
    'm2': 85,
    'habitaciones': 2,
    'banios': 1,
    'planta': 1,
    'tipo_inmueble': 'Piso',
    'estado_conservacion': 'Buen estado',
    'trastero': 0,
    'garaje': 0,
    'piscina': 0,
    'ac': 1,
    'ascensor': 1,
    'terraza': 1,
    'jardin': 1
}])

# Get embedding
    embedding = calculate_new_property_embedding_habitaciones_banios(new_property)
# Convert the embedding to a space-separated string
    embedding_str="["
    embedding_str += ",".join(map(str, embedding[0]))  # Use embedding[0] to access the single row
    embedding_str +="]"
# Print the generated embedding
    print("Generated embedding habitaciones_banios:", embedding_str)

# Get embedding
    embedding = calculate_new_property_embedding_caracteristicas(new_property)
# Convert the embedding to a space-separated string
    embedding_str="["
    embedding_str += ",".join(map(str, embedding[0]))  # Use embedding[0] to access the single row
    embedding_str +="]"
# Print the generated embedding
    print("Generated embedding caracteristicas:", embedding_str)
    

if __name__ == "__main__":
    main()
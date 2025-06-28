import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
from joblib import dump, load
from psycopg2.extras import execute_batch

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
    print (params)
    return params

def fetch_boolean_features_data(engine, chunk_size=10000):
    """Fetch property data from database in chunks"""
    print ('Obteniendo datos de testigos_caracteristicas_view')
    query = """
    SELECT         
        trastero, garaje, piscina, ac, ascensor, terraza, jardin
    FROM testigos_caracteristicas_view

    """
    for chunk in pd.read_sql(query, engine, chunksize=chunk_size):
        yield chunk

def fetch_habitaciones_banios_features_data(engine, chunk_size=10000):
    """Fetch property data from database in chunks"""
    print ('Obteniendo datos de testigos_banios_habitaciones_view')
    query = """
    SELECT         
        habitaciones, banios
    FROM testigos_banios_habitaciones_view
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

def store_pca_model(chunk, embeding_name,numeric_features, categorical_features, boolean_features):
    
    for col in boolean_features:
        chunk[col]=chunk[col].fillna(0).astype(int)
    
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
    
    # Process data in chunks
    print(f"\n Calculado modelo de carateristicas booleanas")
    for chunk in fetch_boolean_features_data(engine, chunk_size=CHUNK_SIZE):
        boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
        store_pca_model(chunk, "caracteristicas",[], [],boolean_features)

    print(f"\n Almacenando modelo de carateristicas booleanas")

    preprocessor_caracteristicas = load('property_preprocessor.caracteristicas.joblib')
    pca_caracteristicas = load('property_pca.caracteristicas.joblib')

    # Calculate embeddings for each row in the chunk
    chunk['embedding_caracteristicas'] = chunk.apply(
        lambda row: calculate_embedding_caracteristicas(
            pd.DataFrame([row]),  # Pass as DataFrame
            preprocessor_caracteristicas,
            pca_caracteristicas
        ),
        axis=1
    )
    
    print(f"\n Actualizando embedding de carateristicas booleanas")
    store_embeddings_caracteristicas(conn, chunk, embedding_column='embedding_caracteristicas', table_name='embedding_dict_caracteristicas')


    update_embeddings_by_features(conn, chunk, embedding_column='embedding_caracteristicas', table_name='testigos')
    update_embeddings_by_features(conn, chunk, embedding_column='embedding_caracteristicas', table_name='testigos_inactivos')
      
    print(f"\n Finalizada actualizacion de embedding de carateristicas booleanas")  
    
    ###############################################################################      
    # Procesar modelo para habitaciones y banios
    ###############################################################################
    
    print(f"\n Calculado modelo de habitaciones y banios")
    
    for chunk in fetch_habitaciones_banios_features_data(engine, chunk_size=CHUNK_SIZE):
        numeric_features = ['habitaciones', 'banios']    
        store_pca_model(chunk, "habitaciones_banios",numeric_features, [],[])

    print(f"\n Almacenando modelo de habitaciones y banios")
    
    preprocessor_caracteristicas = load('property_preprocessor.habitaciones_banios.joblib')
    pca_caracteristicas = load('property_pca.habitaciones_banios.joblib')

    # Calculate embeddings for each row in the chunk
    chunk['embedding_caracteristicas'] = chunk.apply(
        lambda row: calculate_embedding_habitaciones_banios(
            pd.DataFrame([row]),  # Pass as DataFrame
            preprocessor_caracteristicas,
            pca_caracteristicas
        ),
        axis=1
    )
    
    print(f"\n Actualizando embedding de habitaciones y banios")
    store_embeddings_habitaciones_banios(conn, chunk, embedding_column='embedding_caracteristicas', table_name='embedding_dict_habitaciones_banios')
    #update_embeddings_by_habitaciones_banios(conn, chunk, embedding_column='embedding_caracteristicas', table_name='testigos')
    #update_embeddings_by_habitaciones_banios(conn, chunk, embedding_column='embedding_caracteristicas', table_name='testigos_inactivos')
      
    print(f"\n Finalizada actualizacion de embedding de habitaciones y banios")  
      
    
    conn.close()
    
    print("\n Embeddings generados, actualizados y almacenados satisfactoriamente! \n")
    print(f"\n IMPORTANTE: Recuerde actualizar la vista materializada: testigos_materialized_view")  
  
  
  

#####
def store_embeddings_caracteristicas(conn, chunk, embedding_column='embedding_caracteristicas', table_name='embedding_dict_caracteristicas'):
    """
    Insert or update (upsert) embeddings into the embedding dictionary table for boolean features.
    """
    boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
    cursor = conn.cursor()
    data = [
        tuple([bool(row[col]) for col in boolean_features] + [row[embedding_column]])
        for _, row in chunk.iterrows()
    ]
    columns = ', '.join(boolean_features) + ', embedding'
    placeholders = ', '.join(['%s'] * (len(boolean_features) + 1))
    conflict_cols = ', '.join(boolean_features)
    query = f"""
        INSERT INTO {table_name} ({columns})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_cols})
        DO UPDATE SET embedding = EXCLUDED.embedding
    """
    execute_batch(cursor, query, data, page_size=1000)
    conn.commit()
    cursor.close()
    print(f"Embeddings almacenados/actualizados en {table_name} para {len(chunk)} combinaciones.")

def store_embeddings_habitaciones_banios(conn, chunk, embedding_column='embedding_caracteristicas', table_name='embedding_dict_habitaciones_banios'):
    """
    Insert or update (upsert) embeddings into the embedding dictionary table for habitaciones and banios.
    """
    cursor = conn.cursor()
    data = [
        (int(row['habitaciones']), int(row['banios']), row[embedding_column])
        for _, row in chunk.iterrows()
    ]
    columns = 'habitaciones, banios, embedding'
    placeholders = '%s, %s, %s'
    conflict_cols = 'habitaciones, banios'
    query = f"""
        INSERT INTO {table_name} ({columns})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_cols})
        DO UPDATE SET embedding = EXCLUDED.embedding
    """
    execute_batch(cursor, query, data, page_size=1000)
    conn.commit()
    cursor.close()
    print(f"Embeddings almacenados/actualizados en {table_name} para {len(chunk)} combinaciones.")
#####

def update_embeddings_by_features(conn, chunk, embedding_column='embedding_caracteristicas', table_name='testigos'):
    """
    For each row in chunk, update the embedding in the database where all boolean features match.
    """
    boolean_features = ['trastero', 'garaje', 'piscina', 'ac', 'ascensor', 'terraza', 'jardin']
    cursor = conn.cursor()
    idx =1
    for _, row in chunk.iterrows():
        bool_values = [bool(row[col]) for col in boolean_features]
        where_clause = " AND ".join([f"{col} = %s" for col in boolean_features])
        query = f"UPDATE {table_name} SET embedding = %s WHERE {where_clause}"
        params = [row[embedding_column]] + bool_values
        cursor.execute(query, params)
        print(f"Actualizando embedding vector {idx} de {len(chunk)} en tabla '{table_name}' basado en las caracteristicas: {bool_values}")
        idx +=1
        conn.commit()
    cursor.close()
    
def update_embeddings_by_habitaciones_banios(conn, chunk, embedding_column='embedding_caracteristicas', table_name='testigos'):
    """
    For each row in chunk, update the embedding in the database where habitaciones and banios match.
    """
    cursor = conn.cursor()
    idx = 1
    for _, row in chunk.iterrows():
        # Prepare the WHERE clause for habitaciones and banios
        where_clause = "habitaciones = %s AND banios = %s"
        query = f"UPDATE {table_name} SET embedding_habitaciones_banios = %s WHERE {where_clause}"
        params = [row[embedding_column], int(row['habitaciones']), int(row['banios'])]
        cursor.execute(query, params)
        print(f"Actualizando embedding vector {idx} de {len(chunk)} en tabla '{table_name}' basado en habitaciones={row['habitaciones']}, banios={row['banios']}")
        idx += 1
        conn.commit()
    cursor.close()

def calculate_embedding_caracteristicas(property_df,preprocessor_caracteristicas,pca_caracteristicas ):
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

def calculate_embedding_habitaciones_banios(property_df,preprocessor_habitaciones_banios,pca_habitaciones_banios):
    input_df = property_df[['habitaciones', 'banios']]
    processed_data = preprocessor_habitaciones_banios.transform(input_df)
    embedding = pca_habitaciones_banios.transform(processed_data)
    if embedding.shape[1] < 64:
        padded_embedding = np.zeros((1, 64))
        padded_embedding[0, :embedding.shape[1]] = embedding
        return padded_embedding[0].tolist()
    return embedding[0].tolist()

if __name__ == "__main__":
    main()

CREATE TABLE embedding_dict_caracteristicas (
    uuid UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    trastero BOOLEAN NOT NULL,
    garaje BOOLEAN NOT NULL,
    piscina BOOLEAN NOT NULL,
    ac BOOLEAN NOT NULL,
    ascensor BOOLEAN NOT NULL,
    terraza BOOLEAN NOT NULL,
    jardin BOOLEAN NOT NULL,
    embedding VECTOR(64) NOT NULL,
    CONSTRAINT unique_caracteristicas_combination
        UNIQUE (trastero, garaje, piscina, ac, ascensor, terraza, jardin)
);

CREATE INDEX idx_embedding_dict_caracteristicas_features
ON embedding_dict_caracteristicas (trastero, garaje, piscina, ac, ascensor, terraza, jardin);


CREATE TABLE embedding_dict_habitaciones_banios (
    uuid UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    habitaciones INTEGER NOT NULL,
    banios INTEGER NOT NULL,
    embedding VECTOR(64) NOT NULL,
    CONSTRAINT unique_habitaciones_banios_combination
        UNIQUE (habitaciones, banios)
);

CREATE INDEX idx_embedding_dict_habitaciones_banios
ON embedding_dict_habitaciones_banios (habitaciones, banios);


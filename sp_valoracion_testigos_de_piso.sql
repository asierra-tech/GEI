CREATE OR REPLACE FUNCTION _valoracion_testigos_de_piso(
    ref_embedding vector,
    ref_embedding_habitaciones_banios vector,
    ref_precio numeric,
    ref_m2 numeric,
    ref_ubicacion geography,
    ref_habitaciones int,
    ref_banios int,
    ref_planta int,
    ref_ascensor boolean
)
RETURNS TABLE (
    id int,
    precio numeric,
    m2 numeric,
    tipo_inmueble text,
    habitaciones int,
    banios int,
    planta int,
    trastero boolean,
    garaje boolean,
    piscina boolean,
    ac boolean,
    ascensor boolean,
    terraza boolean,
    jardin boolean,
    estado_conservacion text,
    cerramientos text,
    embedding_similarity numeric,
    embedding_similarity_hab_banios numeric,
    distancia_al_inmueble_m numeric,
    combined_score numeric,
    diferencia_porciento_m2 numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
     t.id, 
        t.precio::numeric, -- Explicitly cast to numeric
        t.m2::numeric, -- Explicitly cast to numeric
        t.tipo_inmueble::text, -- Explicitly cast to text
        t.habitaciones,
        t.banios,
        t.planta,
        t.trastero,
        t.garaje,
        t.piscina,
        t.ac,
        t.ascensor,
        t.terraza,
        t.jardin,
        t.estado_conservacion::text,
        t.cerramientos,
        (t.embedding <-> ref_embedding)::numeric AS embedding_similarity, -- Explicit cast to numeric
        (t.embedding_habitaciones_banios <-> ref_embedding_habitaciones_banios)::numeric AS embedding_similarity_hab_banios, -- Explicit cast to numeric
        ST_Distance(t.ubicacion::geography, ref_ubicacion::geography)::numeric AS distancia_al_inmueble_m, -- Explicit cast to numeric
        (
            0.20 * GREATEST(1 - LEAST(t.embedding <-> ref_embedding, 1), 0) +
            0.05 * EXP(-ABS(LN(GREATEST(t.m2::decimal, 0.1)/GREATEST(ref_m2::decimal, 0.1)))) +
            0.30 * GREATEST(1 - LEAST(t.embedding_habitaciones_banios <-> ref_embedding_habitaciones_banios, 1), 0) +
            0.40 * EXP(-ST_Distance(t.ubicacion::geography, ref_ubicacion::geography)/50) +
            0.05 * CASE WHEN t.ascensor = ref_ascensor THEN 1 ELSE 0 END
        )::numeric as combined_score, -- Explicit cast to numeric
        (t.m2::decimal / ref_m2::decimal)::numeric as diferencia_porciento_m2 -- Explicit cast to numeric
    FROM testigos t
    WHERE
        ST_DWithin(t.ubicacion::geography, ref_ubicacion::geography, 1200)
        AND (
            (
                (t.m2::decimal / ref_m2::decimal) BETWEEN 0.80 AND 1.20
                AND t.tipo_inmueble = 'Piso'
                AND t.planta BETWEEN ref_planta - 1 AND ref_planta + 1
                AND t.ascensor = ref_ascensor
            )
            OR (
                (t.m2::decimal / ref_m2::decimal) BETWEEN 0.80 AND 1.20
                AND t.planta = 0
                AND t.tipo_inmueble IN ('Piso')
            )
            OR (
                (t.m2::decimal / ref_m2::decimal) BETWEEN 0.80 AND 1.20
                AND t.ascensor = true
                AND t.tipo_inmueble IN ('Piso', 'Atico', 'Duplex', 'Estudio')
            )
        )
        AND (t.embedding_habitaciones_banios <-> ref_embedding_habitaciones_banios <= 0.6)
        AND (t.tipo_operacion = 'venta')
    ORDER BY distancia_al_inmueble_m ASC
    LIMIT 50;
END;
$$ LANGUAGE plpgsql;
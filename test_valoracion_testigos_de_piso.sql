this query must be called passing parameters instead of WITH, how to achieve it with view or stored procedure? : WITH reference AS (
  SELECT 
    '[1.3147172368960967,-0.6609039685496864,0.3769786739219802,0.9092276221617749,0.3699887625839031,0.5499637179274041,0.6531194120486201,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]'::vector embedding,
    '[-0.5745860042115234,0.24680984547308626,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]'::vector as  embedding_habitaciones_banios,
    precio AS ref_precio,
    85 AS ref_m2,
    ubicacion AS ref_ubicacion,
    2 as ref_habitaciones,
    1 as ref_banios,
    1 as ref_planta,
    true as ref_ascensor
  FROM testigos 
  WHERE id =906150
)
SELECT 
  t.id, 
  t.precio, 
  t.m2,
  t.tipo_inmueble,
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
  t.estado_conservacion,
  t.cerramientos,
  t.embedding <-> r.embedding AS embedding_similarity,
  t.embedding_habitaciones_banios <-> r.embedding_habitaciones_banios AS embedding_similarity_hab_banios,
  
  ST_Distance(t.ubicacion::geography, r.ref_ubicacion::geography) AS distancia_al_inmueble_m,

(
  -- 1. General property embedding similarity (35%)
  0.20 * GREATEST(1 - LEAST(t.embedding <-> r.embedding, 1), 0) +
  
  -- 2. Size comparison using log-normalized ratio (35%)
  0.05 * EXP(-ABS(LN(GREATEST(t.m2::decimal, 0.1)/GREATEST(r.ref_m2::decimal, 0.1)))) +
  
  -- 3. Room/bathroom/floor configuration similarity (15%)
  0.30 * GREATEST(1 - LEAST(
    t.embedding_habitaciones_banios <-> r.embedding_habitaciones_banios, 
    1
  ), 0) +
  
  -- 4. Location proximity with exponential decay (10%)
  0.40 * EXP(-ST_Distance(t.ubicacion::geography, r.ref_ubicacion::geography)/50) +
  
  -- 5. Elevator match bonus (5%)
  0.05 * CASE WHEN t.ascensor = r.ref_ascensor THEN 1 ELSE 0 END
) as combined_score,
   (t.m2::decimal/ref_m2::decimal) as diferencia_porciento_m2
FROM reference r
CROSS JOIN  testigos t
where

ST_DWithin(t.ubicacion::geography, r.ref_ubicacion::geography, 1200) -- Radio de 1 km (en metros)
   
and (
        (
         (t.m2::decimal / r.ref_m2::decimal) BETWEEN 0.80 AND 1.20
            AND t.tipo_inmueble = 'Piso'
            AND t.planta BETWEEN r.ref_planta - 1 AND r.ref_planta + 1
            AND ascensor = r.ref_ascensor
        )
       or
               (
           (t.m2::decimal / r.ref_m2::decimal) BETWEEN 0.80 AND 1.20
            and t.planta = 0
            
            AND t.tipo_inmueble IN ('Piso')
        ) or
        (
           (t.m2::decimal / r.ref_m2::decimal) BETWEEN 0.80 AND 1.20
            
            and t.ascensor = true
            AND t.tipo_inmueble IN ('Piso', 'Atico', 'Duplex', 'Estudio')
        )
    
) 
and ( t.embedding_habitaciones_banios <-> r.embedding_habitaciones_banios <= 0.6)  
and (t.tipo_operacion = 'venta')

ORDER BY  distancia_al_inmueble_m asc

limit 50
; --combined_score desc,

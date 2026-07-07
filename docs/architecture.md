# Arquitectura — Supply Chain Control Tower

Motor de decisión *forecast-to-fulfillment*: predice demanda, decide la
reposición óptima y la valida en un simulador antes de recomendarla.

```
[1] Ingesta (pandas)          M5 Walmart → panel (SKU × tienda × día) + features
        │   contrato de features versionado (configs/feature_contract.yaml)
        ▼
[2] Forecast probabilístico   LightGBM quantile p50/p90/p95 + baseline seasonal-naive
        │   distribución de demanda por SKU-día (no un punto)
        ▼
[3] Optimización              (s,S)/newsvendor cerrado  +  MILP (PuLP/CBC)
        │   política candidata de reposición por SKU
        ▼
[4] Simulación (SimPy)        DES single-echelon + Monte-Carlo (200 réplicas, CRN)
        │   distribución de fill rate / stockouts / costo → política recomendada
        ▼
[5] Servicio + Monitoreo      FastAPI /recommend · Streamlit · Evidently · MLflow
                              GitHub Actions: CI + retrain por drift o caída de KPI
```

Cada flecha es un contrato de datos explícito (parquet con schema testeado).
Una sola función objetivo atraviesa todo: **costo total esperado**
(holding + ordering + stockout) sujeto a nivel de servicio.

## Decisiones de diseño y trade-offs

### 1. Forecast probabilístico, no puntual
La decisión de inventario depende de la **cola** de la demanda: el safety
stock sale de un cuantil (p95), no de la media. Se entrena un LightGBM por
cuantil con pinball loss; los cuantiles se fuerzan a no cruzarse
(monotonía post-hoc). **Truco de leakage:** todos los lags son ≥ 28 días
(= horizonte), así un único modelo directo predice los 28 días sin recursión
y sin ver el futuro. El backtesting es rolling-origin con 3 folds y un assert
explícito de no-leakage. Límite honesto: la cobertura de cuantiles no es
perfecta por construcción → se mide empíricamente en backtesting
(`reports/backtest_summary.json`).

### 2. (s,S) cerrado **y** MILP — los dos, a propósito
- **(s,S)/newsvendor:** `s = μ_L + z·σ_L` con σ implícita del spread p95−p50;
  `S = s + EOQ`. Instantáneo por SKU, escala a millones, pero ignora
  restricciones que acoplan SKUs.
- **MILP (PuLP + CBC):** lot-sizing multi-período con costo fijo de pedido,
  mínimo/múltiplo de pedido, piso de safety stock y **capacidad de bodega
  compartida** entre los 50 SKUs de cada tienda. Óptimo exacto, pero
  O(SKUs×períodos) binarias → escala mal (6 min para 150 SKUs).
- **En producción real:** (s,S) por defecto; MILP solo donde las
  restricciones acoplan (bodega/camión compartido). El repo demuestra ambos y
  los compara **bajo incertidumbre**, no solo en el caso determinista.

### 3. La simulación es el juez
La optimización asume que el forecast es correcto; la simulación mide qué
pasa cuando no lo es. SimPy modela un echelon con revisión diaria, demanda
muestreada de los cuantiles del forecast (negative binomial moment-matched,
sobredispersión típica de retail) y **lead time estocástico Gamma** — las
entregas llegan como eventos en tiempo continuo (órdenes que se cruzan,
llegadas tarde). Las tres políticas enfrentan las **mismas** rutas de demanda
y los mismos streams de lead time (common random numbers) → comparación
justa y de baja varianza. La política recomendada por SKU es la de menor
costo total **esperado** en 200 réplicas.

### 4. Cuándo reentrenar: por señal, no por calendario
Dos triggers explícitos (`src/monitor/retrain_check.py`):
- **Señal estadística:** share de features con drift (Evidently) > 30%.
- **Señal de negocio:** fill rate simulado de la política recomendada < 90%.
Un calendario ciego reentrena tarde (drift rápido) o al pedo (sin drift).
El trigger de negocio es el que importa: drift de features sin impacto en
fill rate no justifica el riesgo de un deploy nuevo.

### 5. Parámetros sintéticos, declarados
M5 no trae lead times ni costos de proveedor. Son **sintéticos, con semilla
fija y documentados** en `configs/params.yaml`: costo unitario = 70% del
precio, holding = 25% anual, penalidad de stockout = 1.5× precio, pedido =
$50 fijo, lead time Gamma(μ=5d, CV=0.3). Nunca datos inventados de una
empresa real.

## Reproducibilidad
- Semilla única (`seed: 42`) → misma corrida, mismos números (test lo verifica).
- `requirements.txt` pinneado; entorno conda `AIEnv`.
- `make pipeline` reconstruye todo desde los CSV crudos de M5.
- Los artefactos procesados (~10 MB) se commitean → CI reentrena sin el raw.

## Límites conocidos
1. **Single-echelon:** un centro por tienda; multi-echelon es extensión futura.
2. **Lead time paramétrico** (Gamma sintética), no datos reales de proveedor.
3. **Cobertura de cuantiles no garantizada** — se reporta la empírica del backtesting.
4. **MILP determinista en demanda** (usa p50 + piso de safety stock); la
   incertidumbre la aporta el simulador, no el MILP (un MILP estocástico
   multi-escenario es extensión natural).
5. El simulador compara políticas **relativamente**; no es la verdad absoluta
   de la cadena real.

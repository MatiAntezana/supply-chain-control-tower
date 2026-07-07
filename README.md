# Supply Chain Control Tower

Sistema de decisión de inventario de punta a punta sobre ventas reales de
Walmart (dataset M5): pronostica la demanda por producto y tienda, calcula la
política de reposición y la valida en un simulador antes de recomendarla.

La idea surge de una limitación conocida de los pipelines de forecasting: un
buen pronóstico no alcanza para decidir cuánto pedir. La decisión de compra
depende de la cola de la distribución de demanda, de los costos de
almacenar/quebrar stock y de la variabilidad del proveedor. Este repo une las
tres piezas.

Demo: https://huggingface.co/spaces/Matycaba2016/supply-chain-control-tower

## Cómo funciona

```
M5 (ventas diarias) -> ingesta -> forecast -> optimización -> simulación -> API/dashboard
                       pandas     LightGBM     (s,S) y MILP    SimPy +        FastAPI +
                                  (cuantiles)  (PuLP/CBC)      Monte-Carlo    Streamlit
```

1. **Ingesta**: convierte los CSV de M5 a un panel producto×tienda×día con
   features (lags, rolling stats, calendario, precio). El esquema está fijado
   en `configs/feature_contract.yaml`; los tests lo validan.
2. **Forecast**: LightGBM con pérdida de cuantil (p50/p90/p95) a 28 días.
   Todos los lags son ≥ 28 días, así el mismo modelo predice el horizonte
   completo sin recursión y sin fuga de información. Se evalúa con backtesting
   rolling-origin contra un baseline seasonal-naive.
3. **Optimización**: dos variantes, a propósito. Una política (s,S) de forma
   cerrada (el punto de pedido sale del cuantil de demanda durante el lead
   time) y un MILP de lot-sizing multi-período que agrega lo que la fórmula no
   ve: capacidad de bodega compartida entre productos, pedido mínimo, múltiplos
   y costo fijo por orden.
4. **Simulación**: un simulador de eventos discretos (SimPy) ejecuta cada
   política contra 200 réplicas de demanda muestreada del forecast y lead
   times Gamma estocásticos. Las políticas comparten las mismas réplicas
   (common random numbers) para que la comparación sea justa. La recomendación
   final por SKU es la política de menor costo esperado.
5. **Servicio y monitoreo**: API FastAPI (`/recommend`), dashboard Streamlit,
   tracking en MLflow, detección de drift con Evidently y un workflow de
   GitHub Actions que reentrena cuando el drift supera el umbral o el fill
   rate simulado cae del piso.

Más detalle de decisiones y trade-offs en `docs/architecture.md`.

## Resultados

Corrida con semilla 42 sobre 150 SKUs (3 tiendas, top-50 por volumen).

Backtesting del forecast (3 folds, horizonte 28 días):

| | LightGBM cuantil | Seasonal-naive |
|---|---|---|
| Cobertura empírica p90 | 0.902 | 0.799 |
| Cobertura empírica p95 | 0.950 | 0.837 |
| Pinball p95 | 0.843 | 1.173 |
| WRMSSE (p50) | 0.669 | 0.677 |

La cobertura es lo que importa acá: si el modelo dice "95% de probabilidad de
vender menos que X", eso ocurre el 95.0% de las veces. El safety stock se
calcula sobre ese cuantil, así que la calibración es la que sostiene todo lo
de abajo.

Políticas bajo incertidumbre (costo esperado por SKU a 28 días, 200 réplicas):

| Política | Fill rate | Costo | vs baseline |
|---|---|---|---|
| Reorder point sin safety stock (baseline) | 93.4% | 105.3 | — |
| (s,S) por cuantiles | 98.6% | 73.1 | −31% |
| MILP | 98.1% | 66.7 | −37% |

Un resultado que no esperaba al arrancar: la curva costo↔servicio
(`reports/cost_service_curve.csv`) baja monótonamente entre 90% y 99% de
nivel de servicio. Con penalidad de stockout de 1.5× el precio (productos
baratos de supermercado), quebrar stock cuesta más que almacenar, y el óptimo
económico queda en servicio alto. Con otra estructura de costos la curva se
da vuelta; el punto del simulador es justamente poder medirlo.

## Correr el proyecto

Requisitos: Python 3.11 (uso un env de conda llamado `AIEnv`), los CSV de M5
en `../m5-forecasting-accuracy/` (se descargan gratis de
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)).

```bash
conda activate AIEnv
pip install -r requirements.txt

make pipeline    # ingesta -> forecast -> optimización -> simulación
make test
make api         # FastAPI en :8000
make dashboard   # Streamlit en :8501
make drift       # reporte Evidently + summary
make retrain-check
```

Con Docker (igual que la demo en Spaces):

```bash
docker build -t sc-control-tower .
docker run -p 7860:7860 sc-control-tower
```

Los artefactos procesados (~10 MB) están commiteados adrede: el CI puede
correr los tests y el ciclo de reentrenamiento sin el dataset crudo (450 MB).

## Estructura

```
configs/            parámetros (semilla, costos, service level) y feature contract
src/ingest/         M5 -> panel + features
src/forecast/       modelo de cuantiles, baseline, backtesting
src/optimize/       (s,S), MILP, economics sintéticos
src/simulate/       DES + Monte-Carlo
src/serve/          API
src/monitor/        drift + trigger de reentrenamiento
app/dashboard.py    Streamlit
tests/              unitarios por capa + integración E2E
```

## Limitaciones

- M5 no trae costos ni lead times de proveedor. Son sintéticos y están
  declarados en `configs/params.yaml` (costo unitario 70% del precio, holding
  25% anual, penalidad de stockout 1.5× precio, lead time Gamma con media 5
  días y CV 0.3). Cambiarlos cambia las conclusiones; por eso están en config
  y no enterrados en el código.
- Single-echelon: cada tienda se abastece directo de un proveedor. No hay
  centro de distribución intermedio.
- El MILP es determinista en demanda (usa el p50 más un piso de safety
  stock); la incertidumbre la evalúa el simulador, no el solver.
- La cobertura de los cuantiles no está garantizada por construcción — por
  eso se mide en backtesting y se reporta.

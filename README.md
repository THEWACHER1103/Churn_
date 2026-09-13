# Predicción de Churn de Clientes

Modelo de Machine Learning para predecir la probabilidad de abandono (churn) de clientes de una empresa de telecomunicaciones, con foco en generar una herramienta de priorización para estrategias de retención.

## Dataset

- **Fuente:** [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) (dataset sintético basado en el Telco Customer Churn dataset).
- **Tamaño:** 594,194 registros, 20 variables, sin valores nulos.
- **Variable objetivo:** `Churn` (binaria) — tasa de abandono ~22.5%.

## Flujo del proyecto

1. **EDA y selección de encoding** — comparación de One-Hot, Target Encoding y WOE (Weight of Evidence) para variables categóricas.
2. **Análisis de multicolinealidad y poder predictivo** — matriz de correlación, VIF e Information Value (IV), antes y después del feature engineering.
3. **Feature Engineering** — variables derivadas: `Sag` (servicios agregados), `Is_New` (antigüedad < 6 meses), `Avg_Charges` (gasto promedio por permanencia), `High_Risk` (combinación de antigüedad, tipo de contrato y método de pago asociada a mayor riesgo).
4. **Competencia de modelos** — Regresión Logística, Random Forest, XGBoost, LightGBM y CatBoost bajo un pipeline con WOE re-ajustado dentro de cada fold de validación cruzada (evitando fuga de información).
5. **Optimización de hiperparámetros** — tuning de XGBoost y LightGBM con Optuna sobre validación cruzada estratificada.
6. **Calibración de probabilidades** — evaluada sobre un conjunto Out-of-Time (OOT), separado del set de test.
7. **Interpretabilidad** — análisis de importancia de variables con SHAP.
8. **Dashboard de ROI** — traducción de las probabilidades del modelo en una herramienta de priorización de campañas de retención.

## Resultados

| Modelo | AUC (Test) | AUC (OOT) | KS (Test) | KS (OOT) |
|---|---|---|---|---|
| XGBoost Baseline | 0.9121 | 0.9135 | 0.6731 | 0.6752 |
| XGBoost + Optuna | 0.9136 | 0.9147 | 0.6744 | 0.6768 |
| **LightGBM + Optuna (modelo final)** | **0.9135** | **0.9147** | **0.6738** | **0.6771** |

XGBoost y LightGBM obtuvieron un desempeño prácticamente idéntico tras el tuning. Se seleccionó **LightGBM** como modelo final por ofrecer una capacidad predictiva equivalente con menor complejidad computacional y tiempos de entrenamiento más eficientes.

## Stack técnico

`Python` · `pandas` / `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `CatBoost` · `Optuna` · `category_encoders` (WOE) · `SHAP` · `statsmodels` (VIF) · `matplotlib` / `seaborn`

## Estructura

```
CHURN.ipynb   # Notebook completo: EDA → Feature Engineering → Modelado → Interpretabilidad → ROI
```

## Notas y limitaciones

- El modelo identifica asociaciones predictivas y niveles de riesgo, **no relaciones causales** — cualquier estrategia de retención derivada debe validarse con experimentación (A/B testing) antes de escalarse.
- El notebook fue desarrollado y entrenado en un entorno con GPU disponible; para ejecutarlo en un entorno sin GPU es necesario ajustar los parámetros `device` de XGBoost y `device_type` de LightGBM a `"cpu"`.
- El umbral de clasificación usado en las métricas de Recall/F1 es 0.5 por defecto; para uso en producción se recomienda optimizarlo en función del costo de negocio (falso positivo vs. falso negativo) en lugar de dejarlo fijo.

## Próximos pasos

- Monitoreo de *data drift* y recalibración periódica del modelo en producción.
- Validación del impacto económico real de las campañas de retención priorizadas con el modelo.
- Explorar la incorporación de variables de comportamiento transaccional adicionales.

---
sidebar_position: 3
---

# Modelo Autorregresivo Recursivo

Definicion...

## Requisitos

- Pandas
- Matplot
- Scikit Learn
- Skforecast

## Instalacón

Ejecutar el siguiente comando en la consola para instalar la librería `Skforescast`

```python
!pip install skforecast
```

## Carga y preparación de los datos

Una vez instalada, se debe importar Pandas y cargar los datos con los que se entrenara el modelo.

```python title="Carga de los datos"
import pandas as pd
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv'
datos = pd.read_csv(url, sep=',')
datos.info()
```

Ejecutamos `data.info()` para verificar que la columna fecha sea de tipo `datetime`.

```bash title="Output"
DatetimeIndex: 195 entries, 1992-04-01 to 2008-06-01
Freq: MS
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   y       195 non-null    float64
 1   exog_1  195 non-null    float64
 2   exog_2  195 non-null    float64
dtypes: float64(3)
```
Se procede a convertir la fecha a `datatime`. También verificaremos que `Pandas` completar los huecos que puedan existir en las columnas con el valor de `null` con el objetivo de asegurar la frecuencia indicada.

```py
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y/%m/%d')
datos = datos.set_index('fecha')
datos = datos.rename(columns={'x': 'y'})
datos = datos.asfreq('MS')
datos = datos.sort_index()
datos.head()
```

<table>
    <tr>
        <th>fecha</th>
        <th>y</th>
        <th>exog_1</th>
        <th>exog_2</th>
    </tr>
    <tr>
        <td>1992-04-01</td>
        <td>0.379808</td>
        <td>0.958792</td>
        <td>1.166029</td>
    </tr>
    <tr>
        <td>1992-05-01</td>
        <td>0.361801</td>
        <td>0.951993</td>
        <td>1.117859</td>
    </tr>
    <tr>
        <td>1992-06-01</td>
        <td>0.410534</td>
        <td>0.952955</td>
        <td>1.067942</td>
    </tr>
    <tr>
        <td>1992-07-01</td>
        <td>0.483389</td>
        <td>0.958078</td>
        <td>1.097376</td>
    </tr>
</table>

## Graficación de los datos

Representación gráfica del conjunto de datos.

```py
datos.plot()
```

![Grafica de datos originales](./img/Autoregresivo%20Recursivo/grafica-datos-original.png)

En el siguiente paso se procede a dividir el conjunto de datos de tal manera que alrededor del 80% de los datos sea destinado a entrenar y el 20% restante sea para probar el modelo.

```py title="separación de los datos entrenamiento - prueba"
steps = 36
datos_train = datos[:-steps]
datos_test  = datos[-steps:]

print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")
```

#### Conjunto de datos para `Entrenamiento`

![Grafica de train](./img/Autoregresivo%20Recursivo/grafica-train.png)

#### Conjunto de datos para `Pruebas`

![Grafica de train](./img/Autoregresivo%20Recursivo/grafica-test.png)

```py
fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
ax.legend();
```

![Grafica de train y test](./img/Autoregresivo%20Recursivo/grafica-train-test-autoregresivo.png)

La sección color naranja es excluida del dataset de entrenamiento para verificar la veracidad de la predicción que arroje el modelo más adelante en este tutorial.

## Entrenamiento del modelo Autorregresivo Recursivo

Como primer paso, se entrenará el modelo `ForecasterAutoreg` a partir de un regresor `RandomForestRegressor` y se establece una ventana temporal de 6 lags. Lo anterior significa que, el modelo utilizará los anteriores 6 meses como predictores.

```python
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 6
             )
forecaster.fit(y=datos_train['y'])
forecaster
```

```bash title="Output"
================= 
ForecasterAutoreg 
================= 
Regressor: RandomForestRegressor(random_state=123) 
Lags: [1 2 3 4 5 6] 
Window size: 6 
Included exogenous: False 
Type of exogenous variable: None 
Exogenous variables names: None 
Training range: [Timestamp('1992-04-01 00:00:00'), Timestamp('2005-06-01 00:00:00')] 
Training index type: DatetimeIndex 
Training index frequency: MS 
Regressor parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': 123, 'verbose': 0, 'warm_start': False} 
Creation date: 2022-05-25 13:48:49 
Last fit date: 2022-05-25 13:48:49 
Skforecast version: 0.4.3 
```

### Predicción

Ahora, una vez entrenado el modelo, pasamos al proceso de predicción donde queremos predecir los 36 meses a futuro.

```python
steps = 36
predicciones = forecaster.predict(steps=steps)
predicciones.head(5)
```

```bash title="Output"
2005-07-01    0.878756
2005-08-01    0.882167
2005-09-01    0.973184
2005-10-01    0.983678
2005-11-01    0.849494
Freq: MS, Name: pred, dtype: float64
```

A continuación, volvemos a graficar el conjunto de datos para comparar los resultados predichos por el modelo con los datos históricos.

```python
fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
```

![Grafica comparativa de train-test-predict](../series-temporales/img/Autoregresivo%20Recursivo/grafica-prediccion.png)

Como se logra observar en la gráfica, el resultado de la predicción no se asemeja demasiado a los datos de prueba. Hasta este punto, es bueno conocer el error que esta arrojando el modelo.

:::caution

Se procede a cuantificar el error que comete el modelo en sus predicciones con la métrica _mean squeared (mse)_.

```python
error_mse = mean_squared_error(
                y_true = datos_test['y'],
                y_pred = predicciones
            )

print(f"Error de test (mse): {error_mse}")
```
Como se observa a continuación, el resultado de la métrica `mse` no muy alto, sin embargo, se puede lograr un error mucho mas bajo, lo cual garantizaria un nivel de confianza mucho mayor a las predicciones.
```bash title="Error Mean Squared Errror"
Error de test (mse): 0.07326833976120374
```
:::



### Ajuste de hiperparámetros (tuning)

El `ForecasterAutoreg` entrenado ha utilizado una ventana temporal de 6 lags y un modelo Random Forest con los hiperparámetros por defecto. Sin embargo, no hay ninguna razón por la que estos valores sean los más adecuados. Para identificar la mejor combinación de lags e hiperparámetros, la librería `Skforecast` dispone de la función `grid_search_forecaster` con la que comparar los resultados obtenidos con cada configuración del modelo.

```python
steps = 36
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # Este valor será remplazado en el grid search
             )

# Lags utilizados como predictores
lags_grid = [10, 20]

# Hiperparámetros del regresor
param_grid = {'n_estimators': [100, 500],
              'max_depth': [3, 5, 10]}

resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos_train['y'],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)*0.5),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False
                   )
```

```bash title="Output"
Number of models compared: 12
loop lags_grid:   0%|                                               | 0/2 [00:00<?, ?it/s]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:00<00:03,  1.40it/s]
loop param_grid:  33%|████████████▋                         | 2/6 [00:03<00:08,  2.18s/it]
loop param_grid:  50%|███████████████████                   | 3/6 [00:04<00:04,  1.52s/it]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:07<00:04,  2.22s/it]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:08<00:01,  1.68s/it]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:12<00:00,  2.26s/it]
loop lags_grid:  50%|███████████████████▌                   | 1/2 [00:12<00:12, 12.06s/it]
loop param_grid:   0%|                                              | 0/6 [00:00<?, ?it/s]
loop param_grid:  17%|██████▎                               | 1/6 [00:00<00:03,  1.42it/s]
loop param_grid:  33%|████████████▋                         | 2/6 [00:04<00:09,  2.26s/it]
loop param_grid:  50%|███████████████████                   | 3/6 [00:04<00:04,  1.56s/it]
loop param_grid:  67%|█████████████████████████▎            | 4/6 [00:08<00:04,  2.31s/it]
loop param_grid:  83%|███████████████████████████████▋      | 5/6 [00:09<00:01,  1.75s/it]
loop param_grid: 100%|██████████████████████████████████████| 6/6 [00:12<00:00,  2.38s/it]
loop lags_grid: 100%|███████████████████████████████████████| 2/2 [00:24<00:00, 12.34s/it]
`Forecaster` refitted using the best-found lags and parameters, and the whole data set: 
  Lags: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20] 
  Parameters: {'max_depth': 3, 'n_estimators': 500}
  Backtesting metric: 0.012836389345193383
```

A continuación, analizamos los resultados arrojados por `grid_search_forecaster`

```
colocar tabla 
```

Los mejores resultados se obtienen si se utiliza una ventana temporal de 20 lags y una configuración de Random Forest {'max_depth': 3, 'n_estimators': 500}.

### Modelo final

Para terminar, volvemos a entrenar el modelo `ForecasterAutoreg` con la configuración optima hallada mediante la validación anterior. 

```
regressor = RandomForestRegressor(max_depth=3, n_estimators=500, random_state=123)
forecaster = ForecasterAutoreg(
                regressor = regressor,
                lags      = 20
             )

forecaster.fit(y=datos_train['y'])
```

Graficamos los resultados de la nueva predicción.

```
predicciones = forecaster.predict(steps=steps)

fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
```

![Grafica del resultado final del modelo](./img/Autoregresivo%20Recursivo/grafica-resultado-modelo.png)

:::tip

Como buena práctica, verificamos una vez más el error producido por el modelo de predicción para saber si los resultados que está alojando el modelo son buenos o no.

```python
error_mse = mean_squared_error(
                y_true = datos_test['y'],
                y_pred = predicciones
            )

print(f"Error de test (mse) {error_mse}")
```

Esta vez, el `error mse` disminuyo considerablemente con respecto al anterior. Paso de un `mse=0.07` a `mse=0.004`. Lo anterior significa que el modelo está realizando predicciones mucho más acertadas que antes gracias al aumento del número de periodos.

```bash title="Nuevo Error Mean Squared Errror"
Error de test (mse) 0.004392699665157793
```
:::
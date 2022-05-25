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

Ejecutar el siguiente comando en la consola para instalar la libreria `Skforescast`

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
Se procede a convertir la fecha a `datatime`. También verificaremos que `Pandas` completar los huecos que puedan existir en las columnas con el valor de `null`  con el objetivo de asegurar la frecuencia indicada.

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

Representación grafica del conjunto de datos.

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

Como primer paso, se entrenara el modelo `ForecasterAutoreg` a partir de un regresor `RandomForestRegressor` y se establece una ventana temporal de 6 lags. Lo anterior significa que, el modelo utilizará los anteriores 6 meses como predictores.

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

A continuacion, volvemos a graficar el conjunto de datos para comparar los resultados predichos por el modelo con los datos historicos.

```python
fig, ax = plt.subplots(figsize=(9, 4))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();
```

![Grafica comparativa de train-test-predict](../series-temporales/img/Autoregresivo%20Recursivo/grafica-prediccion.png)
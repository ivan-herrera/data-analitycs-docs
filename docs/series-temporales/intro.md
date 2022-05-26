---
sidebar_position: 1
---

# Introducción

Los conjuntos de datos que normalmente se exploran en el aprendizaje automático, el tiempo juega un papel importante. Las predicciones son el descubrimiento de hechos que anuncian el futuro, es decir, las predicciones se realizan para encontrar nuevos datos que describan el comportamiento futuro que podría tomar cierta variable de estudio.

## ¿Qué es una Serie de Tiempo?

Una serie de tiempo o también conocida serie temporal es una secuencia de observaciones tomadas secuencialmente en el tiempo.

![Representación de una serie de tiempo](./img/serie%20de%20tiempo.jpeg)

## ¿Por qué hablar de Prediccion de Series Temporales?

La predicción de series temporales es un área muy importante del Machine Learning del que no se habla mucho. En la actualidad, existe una gran cantidad de problemas que involucran el componente tiempo. Sin embargo, es un tema al que no se le da la relevancia correspondiente porque el componente tiempo hace que los problemas sean difíciles de manejar.

Un conjunto de datos común en el Machine Learning representa una colección de observaciones.

Por ejemplo: *Colección de observaciones*

<table>
    <thead>
        <tr>
            <th>Column</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>observation #1</td>
        </tr>
        <tr>
            <td>observation #2</td>
        </tr>
        <tr>
            <td>observation #3</td>
        </tr>
    </tbody>
</table>

Sin embargo, para las series temporales el conjunto de datos que se maneja es diferente. Las series temporales añaden una clara dependencia ordinal explícita entre las observaciones: **la dimensión temporal**. Esta dimensión adicional es una restricción y una estructura que proporciona una fuente adicional de información.

Por ejemplo: *Observaciones de series temporales*

<table>
    <thead>
        <tr>
            <th>Time (index)</th>
            <th>Column</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>time #1</td>
            <td>observation</td>
        </tr>
        <tr>
            <td>time #2</td>
            <td>observation</td>
        </tr>
        <tr>
            <td>time #3</td>
            <td>observation</td>
        </tr>
    </tbody>
</table>

## Ejemplos de Predicciónes de Series Temporales

La aplicación de previsión de series temporales es casi infinita. Existen muchos problemas que pueden ser solucionados con la ayuda de este tema. A continuación, se muestran 10 ejemplos de diferentes disciplinas para que los términos análisis y predicción de series temporales sean más específicos.

- Previsión de la producción de maíz en toneladas por estado cada año.
- Predicción de si un trazado de electroencefalograma en segundos indica que un paciente está sufriendo un ataque o no.
- Predecir el precio de cierre de una acción cada día.
- Predecir la tasa de natalidad en todos los hospitales de una ciudad cada año.
- Predecir las ventas de productos en unidades vendidas cada día para una tienda.
- Predecir el número de pasajeros que pasan por una estación de tren cada día.
- Predecir el desempleo de un estado cada trimestre.
- Predecir la demanda de utilización de un servidor cada hora.
- Predecir el tamaño de la población de conejos en un estado cada temporada de cría.
- Predecir el precio medio de la gasolina en una ciudad cada día.

## Componentes de una Serie de Tiempo

El análisis de series temporales ofrece un conjunto de técnicas para comprender mejor un conjunto de datos. Podemos dividir una serie temporal para que se constituya en 4 partes: 

- **Nivel.** El valor de referencia de la serie si fuera una línea recta.
- **Tendencia.** El comportamiento comúnmente lineal creciente o decreciente de la serie a lo largo del tiempo.
- **Estacionalidad.** La repetición opcional de ciclos de comportamiento a lo largo del tiempo.
- **Ruido.** La variabilidad en las observaciones que no puede ser explicada por el modelo.
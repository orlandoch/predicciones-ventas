# **Predicciones de ventas**
Este proyecto tiene como finalidad crear un modelo de regresión para predecir las ventas de un producto a partir de sus características. Además, durante el proceso se han construido gráficos para entender mejor el conjunto de datos. En las siguientes secciones se mostrará el proceso realizado desde el principio hasta la finalización del proyecto. Una copia de la base de datos se puede descargar desde [aquí](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/).

## **Diccionario de datos**
<table><tbody><tr><td><strong>Nombre de la variable</strong></td><td><strong>Descripción</strong></td></tr><tr><td>Item_Identifier</td><td>Número   de identificación único del producto</td></tr><tr><td>Item_Weight</td><td>Peso del producto</td></tr><tr><td>Item_Fat_Content</td><td>Si el producto es bajo en grasas o regular</td></tr><tr><td>Item_Visibility</td><td>El porcentaje de área total de visualización de todos los productos en la tienda asignados a este producto particular</td></tr><tr><td>Item_Type</td><td>La categoría a la que pertenece el producto</td></tr><tr><td>Item_MRP</td><td>Precio   Máximo Minorista (precio de lista) del producto</td></tr><tr><td>Outlet_Identifier</td><td>Número   de identificación único de la tienda</td></tr><tr><td>Outlet_Establishment_Year</td><td>El año en el que se estableció la tienda</td></tr><tr><td>Outlet_Size</td><td>El tamaño de la tienda en cuanto a la superficie total que cubre</td></tr><tr><td>Outlet_Location_Type</td><td>El tipo de área donde se ubica la tienda</td></tr><tr><td>Outlet_Type</td><td>Si la tienda es un almacén o algún tipo de supermercado</td></tr><tr><td>Item_Outlet_Sales</td><td>Las ventas del producto en la tienda particular. Esta es la variable objetivo que se debe predecir.</td></tr></tbody></table>
<table><tbody><tr><td><strong>Nombre de la variable</strong></td><td><strong>Descripción</strong></td></tr><tr><td>Item_Identifier</td><td>Número   de identificación único del producto</td></tr><tr><td>Item_Weight</td><td>Peso del producto</td></tr><tr><td>Item_Fat_Content</td><td>Si el producto es bajo en grasas o regular</td></tr><tr><td>Item_Visibility</td><td>El porcentaje de área total de visualización de todos los productos en la tienda asignados a este producto particular</td></tr><tr><td>Item_Type</td><td>La categoría a la que pertenece el producto</td></tr><tr><td>Item_MRP</td><td>Precio   Máximo Minorista (precio de lista) del producto</td></tr><tr><td>Outlet_Identifier</td><td>Número   de identificación único de la tienda</td></tr><tr><td>Outlet_Establishment_Year</td><td>El año en el que se estableció la tienda</td></tr><tr><td>Outlet_Size</td><td>El tamaño de la tienda en cuanto a la superficie total que cubre</td></tr><tr><td>Outlet_Location_Type</td><td>El tipo de área donde se ubica la tienda</td></tr><tr><td>Outlet_Type</td><td>Si la tienda es un almacén o algún tipo de supermercado</td></tr><tr><td>Item_Outlet_Sales</td><td>Las ventas del producto en la tienda particular. Esta es la variable objetivo que se debe predecir.
</td></tr></tbody></table>


## Limpieza de datos y datos faltantes
Dentro de la ciencia de datos, una de las primeras tareas a realizar es la limpieza, llenado y eliminación de datos faltantes.

### Limpieza de datos
Ejecutando la línea de código siguiente se evidencia que la columna Item_Fat_Content posee seis valores únicos(['Low Fat' 'Regular' 'low fat' 'LF' 'reg']). Sin embargo, solo dos de ellos (['Low Fat' 'Regular']) se deben mantener, el resto de valores se deben reemplazar con el valor correspondiente. 
```python
print(sp_df['Item_Fat_Content'].unique())
```
Se ha realizado el mismo proceso para todas las variables categóricas con la finalidad de conocer si existen irregularidades similares a las que presenta la variable Item_Fat_Content. Dando como resultado que es la única columna con este tipo de problema en los datos.

Reemplazo de valores:
```python
sp_df['Item_Fat_Content'] = sp_df['Item_Fat_Content'].replace({'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}).to_frame()
```

### Datos faltantes
Para conocer cual es la cantidad de datos faltantes se puede ejecutar la siguiente línea de código:

```python
sp_df.isna().mean()*100
```
Resultado:
```
Item_Identifier               0.000000
Item_Weight                  17.165317
Item_Fat_Content              0.000000
Item_Visibility               0.000000
Item_Type                     0.000000
Item_MRP                      0.000000
Outlet_Identifier             0.000000
Outlet_Establishment_Year     0.000000
Outlet_Size                  28.276428
Outlet_Location_Type          0.000000
Outlet_Type                   0.000000
Item_Outlet_Sales             0.000000
dtype: float64
```

De este modo se observa que las columnas Item_Weight y Outlet_Size continen $17.17\%$ y $28.28\%$ de datos faltantes respectivamente.

Para los datos faltantes de Outlet_Size, primeramente se ha calculado cual es el porcentaje de cada uno de los valores únicos dentro de la culumna, luego se ha completado proporcionalmente los datos faltantes.

```python
y = dfKnn.loc[:,'Item_Outlet_Sales']
inputeDf = dfKnn.drop(columns=['Item_Outlet_Sales'])
inputeDf = pd.concat([dfKnn.select_dtypes(exclude='object'), 
              pd.DataFrame(enc.fit_transform(dfKnn.select_dtypes('object')).toarray().astype(int))],
              axis=1)
```

Los datos faltantes de la columna Item_Weight se han completado con el valor medio de la columna.
```python
dfKnn['Item_Weight'] = dfKnn['Item_Weight'].fillna(dfKnn['Item_Weight'].mean())
```

## Gráficos
A continuación se muestran algunos gráficos

![Cantidad-Locales-Tiempo](https://github.com/orlandoch/predicciones-ventas/raw/main/img/locales_tiempo.png)

![Ventas-Sucursal](https://github.com/orlandoch/predicciones-ventas/raw/main/img/ventas_sucursal.png)

![Precios-Categoría](https://github.com/orlandoch/predicciones-ventas/raw/main/img/caja_bigotes_precios_categoria.png)




## Predicción de ventas con Knn
Utilizando KNN se obtuvo un valor de $R^2_{max}\approx0.47$ 

## Predicción de ventas con Random Forest
Utilizando KNN se obtuvo un valor de $R^2_{max}\approx0.58$, superando a los $\lnapprox0.47$ obtenidos con KNN. Por lo tanto, se seleeciona el modelo generado mediante Random Forest.

![Comparativa-R2](https://github.com/orlandoch/predicciones-ventas/raw/main/img/r2_knn_rf.png)

### Importancia de las variables predictoras
EN el siguiente gráfico se puede observar la importancia de cada una de las variables, es decir, las variables con la mayor capacidad de discriminación.

![Feature-Importance](https://github.com/orlandoch/predicciones-ventas/raw/main/img/feature_importance_rf.png)

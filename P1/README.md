# Ejecutar scripts de la Práctica 1

Para ejecutar los scripts en Python y R es necesario tener los datasets `genero.txt` y `mtcars.txt` en el mismo folder que los scripts.

## Script en R

```sh
$ Rscript analisis.r
```

Este script debería generar 4 gráficas en formato de imagen `png`.

### Output

```sh
$ Rscript analisis.r
[1] "Resumen estadístico de GENERO"
    Gender              Height          Weight
 Length:10000       Min.   :54.26   Min.   : 64.7
 Class :character   1st Qu.:63.51   1st Qu.:135.8
 Mode  :character   Median :66.32   Median :161.2
                    Mean   :66.37   Mean   :161.4
                    3rd Qu.:69.17   3rd Qu.:187.2
                    Max.   :79.00   Max.   :270.0
[1] "Generando boxplot para GENERO"
null device
          1
[1] "Generando gráfica de dispersión para GENERO"
null device
          1



[1] "Resumen estadístico de MTCARS"
     name                mpg             cyl             disp
 Length:32          Min.   :10.40   Min.   :4.000   Min.   : 71.1
 Class :character   1st Qu.:15.43   1st Qu.:4.000   1st Qu.:120.8
 Mode  :character   Median :19.20   Median :6.000   Median :196.3
                    Mean   :20.09   Mean   :6.188   Mean   :230.7
                    3rd Qu.:22.80   3rd Qu.:8.000   3rd Qu.:326.0
                    Max.   :33.90   Max.   :8.000   Max.   :472.0
       hp             drat             wt             qsec
 Min.   : 52.0   Min.   :2.760   Min.   :1.513   Min.   :14.50
 1st Qu.: 96.5   1st Qu.:3.080   1st Qu.:2.581   1st Qu.:16.89
 Median :123.0   Median :3.695   Median :3.325   Median :17.71
 Mean   :146.7   Mean   :3.597   Mean   :3.217   Mean   :17.85
 3rd Qu.:180.0   3rd Qu.:3.920   3rd Qu.:3.610   3rd Qu.:18.90
 Max.   :335.0   Max.   :4.930   Max.   :5.424   Max.   :22.90
       vs               am              gear            carb
 Min.   :0.0000   Min.   :0.0000   Min.   :3.000   Min.   :1.000
 1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:3.000   1st Qu.:2.000
 Median :0.0000   Median :0.0000   Median :4.000   Median :2.000
 Mean   :0.4375   Mean   :0.4062   Mean   :3.688   Mean   :2.812
 3rd Qu.:1.0000   3rd Qu.:1.0000   3rd Qu.:4.000   3rd Qu.:4.000
 Max.   :1.0000   Max.   :1.0000   Max.   :5.000   Max.   :8.000
[1] "Generando boxplot para MTCARS"
null device
          1
[1] "Generando gráficas de dispersión para MTCARS"
null device
          1
```

## Scripts en Python

```sh
$ python genero_regresion.py
$ python genero_gradiente.py
$ python mtcars_regresion.py
$ python mtcars_gradiente.py
```

Es posible también incluir los parámetros `--save-graphs` para generar archivos `.png` con las gráficas incluídas en el reporte, o el parámetro `--display-graphs` para que la librería `matplotlib` muestre las gráficas en una ventana.

### Output del script genero_regresion.py

```sh
$ python genero_regresion.py
 ~ Reading genero.txt and generating train and test sets
 ~ Creating linear regression model
 ~ Testing linear regression model
   → MSE: 149.65234506017003
   → R2: 0.8458290996650849
```

### Output del script genero_gradiente.py

```sh
$ python genero_gradiente.py
 ~ Reading genero.txt and generating train and test sets
 ~ Creating gradient descent model
 ~ Testing gradient descent model
   → MSE: 146.0740416998826
   → R2: 0.86116088288233
```

### Output del script mtcars_regresion.py

```sh
$ python mtcars_regresion.py
 ~ Reading mtcars.txt
 ~ Creating the k-fold cross validation model
 ~ Creating the regression model for every train/test division
   → Partial MSE: 1084.2021044160697 	Partial R2:  0.5838615183889945
   → Partial MSE: 489.0593286910028 	Partial R2:  0.7424343604271374
   → Partial MSE: 1113.585625210933 	Partial R2:  0.818372141719213
   → Partial MSE: 1739.3419823711592 	Partial R2:  0.4639060336353137
   → Partial MSE: 7259.874615138025 	Partial R2:  0.102774573556629
 ~ Calculating final scores
   → Final MSE: 2337.2127311654376
   → Final R2: 0.5422697255454575
```

### Output del script mtcars_gradiente.py

```sh
$ python mtcars_gradiente.py
 ~ Reading mtcars.txt
 ~ Creating the k-fold cross validation model
 ~ Creating the gradient descent model for every train/test division
   → Partial MSE: 1050.4376658286142 	Partial R2:  0.5968209861385975
   → Partial MSE: 447.7300249856181 	Partial R2:  0.7642006532212459
   → Partial MSE: 1124.8361282412384 	Partial R2:  0.8165371640365684
   → Partial MSE: 1570.728654831859 	Partial R2:  0.5158754499195476
   → Partial MSE: 7431.130188262613 	Partial R2:  0.08160962749721401
 ~ Calculating final MSE
   → Final MSE: 2324.9725324299884
   → Final R2: 0.5550087761626348
```

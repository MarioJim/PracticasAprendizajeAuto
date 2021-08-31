# Ejecutar scripts de la Práctica 1

Para ejecutar los scripts de Python es necesario tener los datasets `default.txt` y `genero.txt` en el mismo folder que los scripts.

```sh
$ python default_sklearn.py
$ python default_gradiente.py
$ python genero_sklearn.py
$ python genero_gradiente.py
```

Es posible también incluir los parámetros `--save-graphs` para generar archivos `.png` con las gráficas incluídas en el reporte, o el parámetro `--display-graphs` para que la librería `matplotlib` muestre las gráficas en una ventana.

### Output del script default_sklearn.py

```sh
$ python default_sklearn.py
 ~ Reading default.txt and generating train and test sets
 ~ Creating sklearn's logistic regression model
 ~ Testing sklearn's logistic regression model
   → Accuracy: 0.966
   → Confusion matrix: [[1913, 14], [54, 19]]
```

### Output del script default_gradiente.py

```sh
$ python default_gradiente.py
 ~ Reading default.txt and generating train and test sets
 ~ Creating our logistric regression model with gradient descent
 ~ Testing our logistric regression model with gradient descent
   → Accuracy: 0.964
   → Confusion matrix: [[1928, 0], [72, 0]]
```

### Output del script genero_sklearn.py

```sh
$ python genero_sklearn.py
 ~ Reading genero.txt and generating train and test sets
 ~ Creating sklearn's logistic regression model
 ~ Testing sklearn's logistic regression model
   → Accuracy: 0.918
   → Confusion matrix: [[921, 84], [80, 915]]
```

### Output del script genero_gradiente.py

```sh
$ python genero_gradiente.py
 ~ Reading genero.txt and generating train and test sets
 ~ Creating our logistric regression model with gradient descent
 ~ Testing our logistric regression model with gradient descent
   → Accuracy: 0.9025
   → Confusion matrix: [[925, 104], [91, 880]]
```

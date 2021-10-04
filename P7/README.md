# Ejecutar script de la Práctica 7

Para ejecutar el script `practice7.py` es necesario contar con las librerías _numpy_, _matplotlib_ y _scikit-learn_. Para ejecutar el script es necesario pasar un argumento para seleccionar el modelo de clasificación a utilizar.

```sh
$ python practice7.py linear
$ python practice7.py poly
$ python practice7.py rbf
$ python practice7.py sigmoid
$ python practice7.py logistic
$ python practice7.py knn
$ python practice7.py bayes
```

### Output de la ejecución del comando `python practice7.py linear`

```sh
$ python practice7.py linear
SVM classifier with a linear kernel
Accuracy: 0.9777777777777777
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 34  0  0  0  0  0  0  1  0]
 [ 0  0 36  0  0  0  0  0  0  0]
 [ 0  0  0 29  0  0  0  0  0  0]
 [ 0  0  0  0 30  0  0  0  0  0]
 [ 0  0  0  0  0 39  0  0  0  1]
 [ 0  1  0  0  0  0 43  0  0  0]
 [ 0  0  0  0  1  0  0 38  0  0]
 [ 0  1  1  0  0  0  0  0 37  0]
 [ 0  0  0  1  0  1  0  0  0 39]]
```

### Output de la ejecución del comando `python practice7.py poly`

```sh
$ python practice7.py poly
SVM classifier with a polynomial kernel
Accuracy: 0.9888888888888889
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 35  0  0  0  0  0  0  0  0]
 [ 0  0 36  0  0  0  0  0  0  0]
 [ 0  0  0 29  0  0  0  0  0  0]
 [ 0  0  0  0 30  0  0  0  0  0]
 [ 0  0  0  0  0 39  0  0  0  1]
 [ 0  1  0  0  0  0 43  0  0  0]
 [ 0  0  0  0  0  0  0 39  0  0]
 [ 0  1  0  0  0  0  0  0 38  0]
 [ 0  0  0  0  0  1  0  0  0 40]]
```

### Output de la ejecución del comando `python practice7.py rbf`

```sh
$ python practice7.py rbf
SVM classifier with a Radial Basis Function kernel
Accuracy: 0.9916666666666667
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 35  0  0  0  0  0  0  0  0]
 [ 0  0 36  0  0  0  0  0  0  0]
 [ 0  0  0 29  0  0  0  0  0  0]
 [ 0  0  0  0 30  0  0  0  0  0]
 [ 0  0  0  0  0 39  0  0  0  1]
 [ 0  0  0  0  0  0 44  0  0  0]
 [ 0  0  0  0  0  0  0 39  0  0]
 [ 0  1  0  0  0  0  0  0 38  0]
 [ 0  0  0  0  0  1  0  0  0 40]]
```

### Output de la ejecución del comando `python practice7.py sigmoid`

```sh
$ python practice7.py sigmoid
SVM classifier with a sigmoid kernel
Accuracy: 0.9138888888888889
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 28  0  0  1  0  2  2  1  1]
 [ 0  2 33  0  0  0  0  0  0  1]
 [ 0  0  0 27  0  0  0  0  2  0]
 [ 0  0  0  0 29  0  0  1  0  0]
 [ 0  0  0  0  0 39  0  0  0  1]
 [ 0  1  0  0  2  0 41  0  0  0]
 [ 1  2  0  0  0  0  0 36  0  0]
 [ 0  5  0  2  0  0  0  0 32  0]
 [ 0  0  0  0  0  1  0  2  1 37]]
```

### Output de la ejecución del comando `python practice7.py logistic`

```sh
$ python practice7.py logistic
Logistic regression classifier
Accuracy: 0.9472222222222222
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 31  0  0  0  0  1  0  3  0]
 [ 0  0 34  2  0  0  0  0  0  0]
 [ 0  0  0 28  0  1  0  0  0  0]
 [ 0  0  0  0 30  0  0  0  0  0]
 [ 0  0  0  0  0 39  0  0  0  1]
 [ 0  1  0  0  0  0 43  0  0  0]
 [ 0  0  0  0  2  0  0 37  0  0]
 [ 0  3  1  0  0  0  0  0 34  1]
 [ 0  0  0  1  0  1  0  0  1 38]]
```

### Output de la ejecución del comando `python practice7.py knn`

```sh
$ python practice7.py knn
k-Nearest Neighbors classifier
Accuracy: 0.9888888888888889
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 35  0  0  0  0  0  0  0  0]
 [ 0  0 35  1  0  0  0  0  0  0]
 [ 0  0  0 29  0  0  0  0  0  0]
 [ 0  0  0  0 30  0  0  0  0  0]
 [ 0  0  0  0  0 39  0  0  0  1]
 [ 0  0  0  0  0  0 44  0  0  0]
 [ 0  0  0  0  0  0  0 39  0  0]
 [ 0  0  0  0  0  0  0  0 39  0]
 [ 0  0  0  1  0  1  0  0  0 39]]
```

### Output de la ejecución del comando `python practice7.py bayes`

```sh
$ python practice7.py bayes
Naive Bayes classifier
Accuracy: 0.8444444444444444
Confusion matrix:
[[27  0  0  0  0  0  0  0  0  0]
 [ 0 20  8  0  0  0  0  0  5  2]
 [ 0  1 29  4  0  0  0  1  1  0]
 [ 0  0  1 25  0  0  0  0  1  2]
 [ 0  0  0  0 28  0  0  2  0  0]
 [ 1  1  0  1  0 32  0  0  0  5]
 [ 0  1  0  0  1  1 41  0  0  0]
 [ 0  0  0  0  0  0  0 39  0  0]
 [ 0  4  0  1  0  0  0  1 31  2]
 [ 0  0  0  3  2  1  0  2  1 32]]
```

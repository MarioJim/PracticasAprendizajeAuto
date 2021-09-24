# Ejecutar scripts de la Práctica 3

Para ejecutar los scripts de Python es necesario tener los datasets `default.txt` y `genero.txt` en el mismo folder que los scripts.

```sh
$ python default.py
$ python default.py
```

Es posible también incluir los parámetros `--save-graphs` para generar archivos `.png` con las gráficas incluidas en el reporte, o el parámetro `--display-graphs` para que la librería `matplotlib` muestre las gráficas en una ventana.

### Output del script default.py

```sh
$ python default.py
 ~ Reading default.txt and generating train and test sets
 ~ Creating and testing the k-NN models for different values
   → neighbors   accuracy       confusion matrix
   →     1        0.949     [[1872, 54], [48, 26]]
   →     2        0.9655    [[1920, 6], [63, 11]]
   →     3        0.9635    [[1923, 3], [70, 4]]
   →     5        0.963     [[1925, 1], [73, 1]]
   →    10        0.963     [[1926, 0], [74, 0]]
   →    15        0.963     [[1926, 0], [74, 0]]
   →    20        0.963     [[1926, 0], [74, 0]]
   →    50        0.963     [[1926, 0], [74, 0]]
   →    75        0.963     [[1926, 0], [74, 0]]
   →    100       0.963     [[1926, 0], [74, 0]]


 ~ Creating the logistic regression model
 ~ Testing the logistic regression model
   → Accuracy: 0.965
   → Confusion matrix: [[1915, 11], [59, 15]]
```

### Output del script genero.py

```sh
$ python genero.py
 ~ Reading genero.txt and generating train and test sets
 ~ Creating and testing the k-NN models for different values
   → neighbors   accuracy       confusion matrix
   →     1        0.8845    [[896, 116], [115, 873]]
   →     2        0.8795    [[945, 67], [174, 814]]
   →     3        0.867     [[966, 46], [220, 768]]
   →     5        0.8395    [[981, 31], [290, 698]]
   →    10        0.807     [[991, 21], [365, 623]]
   →    15        0.7875    [[996, 16], [409, 579]]
   →    20        0.7725    [[999, 13], [442, 546]]
   →    50        0.722     [[1008, 4], [552, 436]]
   →    75        0.6995    [[1010, 2], [599, 389]]
   →    100       0.6775    [[1010, 2], [643, 345]]

   → Confusion matrix: [[896, 116], [115, 873]]


 ~ Creating the logistic regression model
 ~ Testing the logistic regression model
   → Accuracy: 0.9225
   → Confusion matrix: [[937, 75], [80, 908]]
```

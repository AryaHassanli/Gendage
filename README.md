# Gendage

Gender Recognition & Age Estimation

``` shell
usage: main.py [-h] [-G] [--dataset dataset] [--feature feature] [--splitSize splitSize splitSize splitSize] [--batchSize batchSize] [--epochs num of epochs] function

positional arguments:
  function              the function to execute. choices: train

optional arguments:
  -h, --help            show this help message and exit
  -G, --gradient        to run on gradient (default: False)
  --dataset dataset     select the dataset: UTKFace, AgeDB (default: UTKFace)
  --feature feature     select the feature: age, gender (default: gender)
  --splitSize splitSize splitSize splitSize
                        specify the train, validation and test size (default: [0.7, 0.2, 0.1])
  --batchSize batchSize
                        set the batch size (default: 15)
  --epochs num of epochs
                        set the number of epochs (default: 10)
```

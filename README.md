# DADA: Deep Adversarial Data Augmentation


----------


DADA: Deep Adversarial Data Augmentation for Extremely Low Data Regime Classification

See: https://arxiv.org/abs/1809.00981
![image](https://github.com/SchafferZhang/DADA/blob/master/imgs/1.png)
CIFAR10, SVHN experiments in the cifar10-svhn folder
CIFAR100 experiments in the cifar100 folder
Tumor classification experiments in the folder DDSM


----------


## Prerequisite
- Python 2.7 or Python 3.3+
- Theano >= 0.9.0
- Lasagne >= 0.1.0
- Keras 

All preprocessed data have been placed into the datasets folder, if you want to test on your own data set, just follow the format of the provided. The results including the training stage and generation stage will be found in the results folder. 

## Training stage
For CIFAR10 experiments, train a model with n = 200 (200 samples per class), batch size = 100, with traditional data augmentation technique

    Python train_cifar_svhn.py --batch_size 100 --dataset cifar10 --count 200 --aug
For CIFAR100 experiments, train a model with n = 200 (200 samples per class), batch size = 100, with traditional data augmentation technique

    Python train_cifar100.py --batch_size 100 --dataset cifar100 --count 200 --aug

For SVHN experiments, train a model with n = 200 (200 samples per class), batch size = 100

    Python train_cifar_svhn.py --batch_size 100 --dataset svhn --count 200

For tumor classification experiments, train a model with traditional data augmentation technique, batch size = 16

    Python train_ddsm.py --batch_size 16 --dataset DDSM --aug

## Generation and basic classifier
run the generation.py and classification.py

## Results
### Emotion Recognition from Facial Expressions: Comparison with Transfer Learning (TL)
Transfer learning is often an effective choice for problems short of training data. But their effectiveness is limited when there are domain mismatches. Our proposed method shows competitive performance compared to other GAN-based augmentation techniques. 

|TL (baseline)|vanilla GAN|Improved GAN| DADA|
|--------|:------:|:------:|:------:|
|82.86%|83.27%|84.03%|**85.71%**|

### Brain Signal Classification: No Domain Knowledge Can be Specified for Augmentation
The following table shows the performance advantage of DADA over the competitive CNN-SAE method in all nine subjects, with an average accuracy margin of 1.7 percent.

|Method |  Sub. 1 |  Sub. 2  | Sub. 3  | Sub. 4  | Sub. 5 | Sub. 6 | Sub. 7 | Sub. 8 | Sub. 9 | **Average**|
|-------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
 |SVM | 71.8 | 64.5 | 69.3 | 93.0 | 77.5 | 72.5 | 68.0 | 69.8 | 65.0 | 72.4 |
 |CNN | 74.5 | 64.3 | 71.8 | 94.5 | 79.5 | 75.0 | 70.5 | 71.8 | 71.0 | 74.8 |
 |CNN-SAE | 76.0 | 65.8 | 75.3 | 95.3 | 83.0 | 79.5 | 74.5 | 75.3 | 75.3  | 77.6| 
 |DADA | **76.6** | **66.8** | **75.6** | **96.5** | **83.2** | **80.2** | **77.0** | **78.6** | **79.6** | **79.3**|

### Tumor Classification: Comparison with Other Learning-based Augmentation
The results shows the clear advantage of our approaches.

|Method|Acc|
|--------|:--------:|
|Tanda (MF)|0.5990|
|Tanda (LSTM)|0.6270|
|DADA|0.6196|
|DADA_augmented|**0.6549**|
### CIFAR10
Our DADA model shows an competitive results on CIFAR10 data set
![image](https://github.com/SchafferZhang/DADA/blob/master/imgs/2.png)

## Vision
![image](https://github.com/SchafferZhang/DADA/blob/master/imgs/3.png)
![image](https://github.com/SchafferZhang/DADA/blob/master/imgs/4.png)

## Acknowledgments
We heavily borrow the code from [Improved-GAN](https://github.com/openai/improved-gan)


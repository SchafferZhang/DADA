# DADA: Deep Adversarial Data Augmentation


----------


DADA: Deep Adversarial Data Augmentation for Extremely Low Data Regime Classification
![image](https://github.com/SchafferZhang/DADA/tree/master/imgs/1.png)
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
Our DADA model shows an competitive results on CIFAR10 data set
![image](https://github.com/SchafferZhang/DADA/tree/master/imgs/2.png)

## Vision
![image](https://github.com/SchafferZhang/DADA/tree/master/imgs/3.png)
![image](https://github.com/SchafferZhang/DADA/tree/master/imgs/4.png)

## Acknowledgments
We heavily borrow the code from [Improved-GAN](https://github.com/openai/improved-gan)


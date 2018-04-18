# ImageClassifier


The code uses Keras & Tensorflow to train a convolutional neural network on a labeled dataset of cats and dogs. Then, it will be able to classify novel cats and dogs pretty well.
By changing the model architecture it can be used to classify multiple labels.


Download Data from : https://www.kaggle.com/c/dogs-vs-cats/data
For now split the "train" data into "train" and "validation". Ratio 80:20
Also keep the in directories.
  train cats data into "/data/train/cat/"
  train dogs data into "/data/train/dog/"
  validation cats data into "/data/validation/cat/"
  validation dogs data into "/data/validation/dog/"


Language
============

* Python 3

Usage
============

Run the script in terminal via

python cat_vs_dog.py 0    # for small convnet
python cat_vs_dog.py 1    # for Vgg16 convnet


Credits
============
Credits to [Siraj](https://github.com/llSourcell)
I've restructured and added few methods to meet the requirements that suited my needs.

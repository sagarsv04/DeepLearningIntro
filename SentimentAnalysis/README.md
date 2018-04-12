# SentimentAnalysis


The code uses different dataset and train our neural network to compute loss values.
I'm using tflearn library to build and train our neural network.

Note
============
* tensorflow = 1.6.0
* tflearn = 0.3.2

Using tflearn's to_categorical method on my labels: It generate same results for all classes
As the method takes only single value and not single value array or list.
I have implemented a method to perform same operation on my labels.


Language
============

* Python 3


Usage
============

Run the script in terminal via

python sentiment_analysis.py


Credits
============
Credits to [Siraj](https://github.com/llSourcell)
I've restructured and added few methods to meet the requirements that suited my needs.

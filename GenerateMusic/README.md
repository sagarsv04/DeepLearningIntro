# GenerateMusic

This is the code for generating music using LSTM network.
It uses Keras & Theano, two deep learning libraries, to generate jazz music. Specifically, it builds a two-layer LSTM, learning from the given MIDI file.


Language
============

* Python 3


Usage
============

Run on CPU with command:  

    python generator.py [# of epochs]

Run on GPU with command:

    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python generator.py [# of epochs]


Credits
============
Credits to [Siraj](https://github.com/llSourcell)\n

I've restructured and added few methods to meet the requirements that suited my needs.

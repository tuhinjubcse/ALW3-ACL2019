# ALW3-ACL2019

Repo for our paper

# Pay "Attention'' to your Context when Classifying Abusive Language

# Third Proceedings of the Abusive Language Workshop, ACL 2019 , Florence Italy


Download twitter datasets and preprocess using Ekphrasis
There is some skeleton code on how to use Ekphrasis
You can alter based on your file and how you have the tweet and label corresponding to it


textClassifierHATT.py has code for both Self and Context Attention


Self Attention ----> Class Name Attention

Context Attention ---> Class Name AttLayer



You can edit when adding in the model

For running you have to use python2 and Theano as BACKEND to run the code
Use command

KERAS_BACKEND=theano python2 textClassifierHATT.py

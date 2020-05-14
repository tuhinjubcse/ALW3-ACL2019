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


If you use our code or idea, please cite us

          @inproceedings{chakrabarty-etal-2019-pay,
              title = "Pay {``}Attention{''} to your Context when Classifying Abusive Language",
              author = "Chakrabarty, Tuhin  and
                Gupta, Kilol  and
                Muresan, Smaranda",
              booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
              month = aug,
              year = "2019",
              address = "Florence, Italy",
              publisher = "Association for Computational Linguistics",
              url = "https://www.aclweb.org/anthology/W19-3508",
              doi = "10.18653/v1/W19-3508",
              pages = "70--79",
              abstract = "The goal of any social media platform is to facilitate healthy and meaningful interactions among its users. But more often than not, it has been found that it becomes an avenue for wanton attacks. We propose an experimental study that has three aims: 1) to provide us with a deeper understanding of current data sets that focus on different types of abusive language, which are sometimes overlapping (racism, sexism, hate speech, offensive language, and personal attacks); 2) to investigate what type of attention mechanism (contextual vs. self-attention) is better for abusive language detection using deep learning architectures; and 3) to investigate whether stacked architectures provide an advantage over simple architectures for this task.",
          }

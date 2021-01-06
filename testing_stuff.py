import numpy as np
l = ['excellent film', 'really poorly executed']
words = {word for review in l for word in review.split(' ')}
for word in words:
    print(word)
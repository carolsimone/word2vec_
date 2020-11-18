# word2vec_
_Creating Google's word2vec algorithm from scratch._

**word2vec** literally translates word to vectors, this allows to calculate a semantic distance between words, aka **Euclidean Distance**. Words such as _King_ and _Queen_ have a small distance between each other (as original word2vec documentation outlines), but _Queen_ and the word _tea_ (yeah, I'm living in the UK right now, I had to pick that word up) does not have anything in common, hence the Euclidean Distance should be larger. In this case our hidden layers has two neurons, `embeddings_size = 2`, and **the weights of our trained model between input layer and hidden layer are going to create the words embedding vectors**. This is a key concept of word2vec.

Let's try to implement an algorithm that is able to create this relationships between words.

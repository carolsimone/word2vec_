# word2vec_
<font color="green">_Creating Google's word2vec algorithm from scratch._</font>

**word2vec** literally translates word to vectors, this allows to calculate a semantic distance between words aka **Euclidean Distance**. _King_ and _Queen_ have a small distance between each other (as original word2vec documentation outlines), but _Queen_ and the word _tea_ does not have anything in common, hence the Euclidean Distance should be larger.

Let's try to implement an algorithm that is able to create this relationships between words.

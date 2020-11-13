# word2vec_
<font color="green">_Creating Google's word2vec algorithm from scratch._</font>

**word2vec** literally translates word to vectors, this allows to calculate a semantic distance between words, aka **Euclidean Distance**. Words such as _King_ and _Queen_ have a small distance between each other (as original word2vec documentation outlines), but _Queen_ and the word _tea_ (yeah, I'm living in the UK right now, I had to pick that word up) does not have anything in common, hence the Euclidean Distance should be larger.

Let's try to implement an algorithm that is able to create this relationships between words.

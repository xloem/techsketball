Hi, I'm kinda crazy.

I thought it would be fun to try to train transformer models to reverse engineer.

I have trouble doing things, and don't really know how to do that, so just a few bursts are likely to appear here.

---
I recommend using jax/flax and google cloud, because google has a TPU research program that could be leveraged for compute
once a setup is designed.

Google's systems are accessible on the web at https://colab.research.google.com/

I propose providing the data to be reversed as raw embeddings, rather than token ids, because many of them may have arithmetic relationships with each other that could be lost in the tokenization process.

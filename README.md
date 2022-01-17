Hi, I'm kinda crazy.

I thought it would be fun to try to train transformer models to reverse engineer.

I have trouble doing things, and don't really know how to do that, so just a few bursts are likely to appear here.

---
I recommend using jax/flax and google cloud, because google has a TPU research program free trail w/ application that could be leveraged for compute
once a setup is designed.

I do NOT normally recommend using google cloud, because your muscle contraction timing will be harvested by javascript to guide your behavior for some of the world's largest marketing computers.

Google's systems are accessible on the web to the public for use at https://colab.research.google.com/

Here is information on using T5, a model framework that has been successful at language translation: https://huggingface.co/docs/transformers/model_doc/t5

Here is an example colab notebook for training a transformer model on tpus: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/causal_language_modeling_flax.ipynb .  If followed, the train_step function would be updated per t5 usage, e.g. using the loss function mentioned in the t5 link above.

Here is a paper on possibly dropping training memory requirements to their square root.  I'm not sure if I understand things right, but this might mean that input data chunks could be much much longer: https://arxiv.org/abs/2112.05682

I propose ensuring the data to be reversed can have its numeric values preserved by the model, because many of them may have arithmetic relationships with each other that could be lost in the tokenization process.  This may mean skipping tokenization, and possibly embedding.  It looks like the simplest way to consider skipping embedding could be to simply alter the embedding weights to have a desired effect (e.g. set an axis to a linear range).

Here's a link to google's free tpu research program: https://sites.research.google/trc/about/

---
Pending concerns:

- A new tokenization strategy is needed, to preserve the information of embedded strings.  This will significantly simplify the training.

Hi, I'm kinda crazy.

I thought it would be fun to try to train transformer models to reverse engineer.

I have trouble doing things, and don't really know how to do that, so just a few bursts are likely to appear here.

---
I recommend using jax/flax and google cloud, because google has a TPU research program free trail w/ application that could be leveraged for compute
once a setup is designed.

Google's systems are accessible on the web at https://colab.research.google.com/

Here is information on using T5, a model framework that has been successful at language translation: https://huggingface.co/docs/transformers/model_doc/t5

Here is a paper on possibly dropping training memory requirements to their square root.  I'm not sure if I understand things right, but this might mean that input data chunks could be much much longer: https://arxiv.org/abs/2112.05682

I propose providing the data to be reversed as raw embeddings, rather than token ids, because many of them may have arithmetic relationships with each other that could be lost in the tokenization process.

----
```
# model import sketch
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
!pip install deepspeed transformers

from transformers import FlaxT5ForConditionalGeneration
startng_model_path = 'bigscience/T0pp'

model = FlaxT5ForConditionalGeneration.from_pretrained(model_path)
# explore model properties in interactive environment to find embeddings matrix or view source code of forward function?
# can raw embeddings be passed to this model class?
```

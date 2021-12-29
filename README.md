Hi, I'm kinda crazy.

I thought it would be fun to try to train transformer models to reverse engineer.

I have trouble doing things, and don't really know how to do that, so just a few bursts are likely to appear here.

---
I recommend using jax/flax and google cloud, because google has a TPU research program free trail w/ application that could be leveraged for compute
once a setup is designed.

Google's systems are accessible on the web to the public for use at https://colab.research.google.com/

Here is information on using T5, a model framework that has been successful at language translation: https://huggingface.co/docs/transformers/model_doc/t5

Here is a paper on possibly dropping training memory requirements to their square root.  I'm not sure if I understand things right, but this might mean that input data chunks could be much much longer: https://arxiv.org/abs/2112.05682

I propose ensuring the data to be reversed can have its numeric values preserved by the model, because many of them may have arithmetic relationships with each other that could be lost in the tokenization process.  This may mean skipping tokenization, and possibly embedding.  It looks like the simplest way to consider skipping embedding could be to simply alter the embedding weights to have a desired effect (e.g. replace a plane with the identity matrix).

----
```
# model import sketch
!pip3 install deepspeed
!pip3 install transformers
!pip3 install flax

from transformers import FlaxT5ForConditionalGeneration
startng_model_path = 'bigscience/T0pp'

model = FlaxT5ForConditionalGeneration.from_pretrained(model_path)
# does not provide for raw embeddings but matrix is available: where?
```

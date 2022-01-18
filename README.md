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
Recorded issues:

Overall end goal: Make it easy for humans to duplicate and comprehend complex objects, even if it is very slow to do so.  i.e. utility.

Possible next goal: Verify an approach works.

A good next step might be speed, likely low-end cloud tpus as being more generally reusable, to approach verification more rapidly.

Speed:
- [ ] I have not set up training for google cloud tpus or for decentralization.
- [ ] Training a new tokenizer significantly slows training.  For preserving bytes, it would be better to reuse the existing tokens the model has.

Indications of Error:
- [ ] The bytestokenizer does not preserve data perfectly, which could cause small problems.  For
 example, "  '" may be turned into " '" when detokenized.  The behavior may be inside sentencepiece.
- [ ] Using the custom tokenizer, less data appears loaded, implying some is lengthened.  This shouldn't be the case.

Effectiveness:
- [ ] The notebook does training, but does not automatically store its model anywhere, which would be a great addition to not lose time spent training.
- [ ] When using a new tokenizer, should the embeddings be randomized before retraining?
- [ ] The tokenizer is trained on single-language data which reduces reusability of the model.
- [ ] Have not yet added fixing an axis of embeddings to be proportional to byte values.

Utility of use:
- [ ] The tokenizer is wrapped in a custom class, which makes the generated model harder to use.  The behaviors of the custom class could be simplified into the embedding layer of the model.
- [ ] The example of pyc->py is not useful.  A different source language would be more useful.
- [ ] There is no supporting code for using a model yet.  It would be helpful to have maybe a class that retrieves a model and uses it.

Utility of development:
- [ ] The training code is copypasta.
- [ ] There is no packaging.

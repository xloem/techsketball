# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/xloem/techsketball/blob/main/model_import_sketch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + colab={"base_uri": "https://localhost:8080/"} id="nH1Ld_vd9wyx" outputId="a1c96dd3-4cd0-473f-a8f3-2fbebf303cc7"
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
# #!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip3 install deepspeed
# !pip3 install transformers
# !pip3 install flax
# !pip3 install sentencepiece
# !git clone https://github.com/xloem/techsketball && ln -s techsketball/* .

# + colab={"base_uri": "https://localhost:8080/"} id="mTGBjrXoX2eS" outputId="ce5bf3c5-0f0f-42bf-e43f-909f2b8c7917"
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
jax.local_devices()

# + id="qhJhFJfQAOXG"
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration
starting_model_path = 't5-base'#'bigscience/T0pp'

tokenizer = T5Tokenizer.from_pretrained(starting_model_path) # only for source, not for binary
model = FlaxT5ForConditionalGeneration.from_pretrained(starting_model_path)

# + id="QnyDTDt_f1fE"
from find_pycode import pair_finder
data_tuples = [*pair_finder(globals())] # tuples of (bytes, str)

# + id="TrRadfeAhZSd" outputId="21397f28-7958-464e-f99e-d89b9e861cd1" colab={"base_uri": "https://localhost:8080/", "height": 397}
input_width = max((len(bytes) for bytes, str in data_tuples))
print('converting byteslist to jax array .. is there a faster way to do this?')
data_input_ids = jax.numpy.array([
    jax.numpy.pad(jax.numpy.array([*bytes], dtype=jax.numpy.uint8), input_width) for bytes, str in data_tuples
])
print('converted')
data_labels = [str for bytes, str in data_tuples]


# + id="sdJ1Ek3-_j37"
#import deepspeed
#cmd_args = None
#model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                     model=model,
#                                                     model_parameters=params)

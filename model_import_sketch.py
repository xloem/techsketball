# ---
# jupyter:
#   jupytext:
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

# + colab={"base_uri": "https://localhost:8080/"} id="nH1Ld_vd9wyx" outputId="9d084e9b-c8b2-493c-ea3e-f2a288d43c44"
#[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

starting_model_path = 't5-base'#'bigscience/T0pp'
input_width = 512


# #!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip3 install deepspeed
# !pip3 install transformers
# !pip3 install flax
# !pip3 install sentencepiece
# !git clone https://github.com/xloem/techsketball && ln -s techsketball/* .

# + id="mTGBjrXoX2eS"
#import jax.tools.colab_tpu
#jax.tools.colab_tpu.setup_tpu()
#jax.local_devices()

from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(starting_model_path) # only for source, not for binary
model = FlaxT5ForConditionalGeneration.from_pretrained(starting_model_path)

# + colab={"base_uri": "https://localhost:8080/"} id="QnyDTDt_f1fE" outputId="36a849f6-4bb4-45f1-e8ef-dd8edf142fa5"
from find_pycode import pair_finder
print('generating training data ...')
data_tuples = [
  (bytes, str)
  for bytes, str
  in pair_finder(globals()) # finds code examples
  if len(bytes) <= input_width
]

# + id="TrRadfeAhZSd"
import jax, numpy as np
data_input_ids = jax.numpy.stack([
  np.frombuffer(bytes.ljust(input_width, b'\0'), dtype=np.uint8)
  for bytes, str in data_tuples
])
data_labels = [str for bytes, str in data_tuples]


# + id="sdJ1Ek3-_j37"
#import deepspeed
#cmd_args = None
#model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                     model=model,
#                                                     model_parameters=params)

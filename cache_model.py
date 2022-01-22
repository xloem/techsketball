# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/xloem/techsketball/blob/main/model_import_sketch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="nH1Ld_vd9wyx" outputId="8d6c59c9-8c72-4176-a8ee-ced7748486a8" colab={"base_uri": "https://localhost:8080/"}
#[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

import jax

starting_model_path = 'baffo32/pyc2py_alpha' #'t5-base'#'t5-small'#'bigscience/T0pp'
train_tokenizer = False # baffo32/pyc2py_alpha already has a tokenizer that can output bytes

input_width = 512
# these are not t5 parameters?
train_batch_size = 8 # small for notebook
per_device_batch_size = train_batch_size // jax.device_count()
num_epochs = 1
training_seed = 0
learning_rate = 0.0001#3e-4
logging_steps = 32


# #!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip3 install deepspeed
# !pip3 install transformers
# !pip3 install flax
# !pip3 install sentencepiece
# !git clone https://github.com/xloem/techsketball && ln -s techsketball/* .
# !apt-get install git-lfs
# !git config --global user.email johndoe@example.com
# !git config --global user.name 'John Doe'
# !git config --global credential.helper store

# + colab={"base_uri": "https://localhost:8080/"} id="mTGBjrXoX2eS" outputId="e0308b1c-db0d-4c8a-87f6-6e38a44d0f49"
import jax.tools.colab_tpu
import jaxlib
import os
if 'COLAB_TPU_ADDR' in os.environ:
  jax.tools.colab_tpu.setup_tpu()
  backend = 'tpu'
else:
  backend = jax.default_backend()
import tensorflow as tf
# !nvidia-smi
jax.local_devices()

# + id="qhJhFJfQAOXG" colab={"base_uri": "https://localhost:8080/", "height": 150, "referenced_widgets": ["bc1a718ac99041f2b8b3d829bb836c69", "88d17fd4ed5d4f958ac64549e4a4158e", "cd4d0e0eba7141739a6b943ae6bf6d27", "98ec7dfcd078422f893d2352e7dcb824", "d96fb68f26da47d186d09d5a0f8a251a", "4737827c8fa74c68bfcbfd19662d51b9", "594bca751e6b4867817b6eb29b4bba32", "00a4ea355c8a4cdc86505b7bf21bd0a1", "ec1014ce01544b6eb9d6ac26eaaa9a43", "45bd21affa2d4623b7e730678fb81454", "d2052a0cb77e431497eb34b3c73311fb", "8d3269f85c054fbcaef78c25421f5f2c", "6bcd5d42d8114209bc92d2c2b09296b2", "98ee691ccc6d4c98bc847e77fef7a0e3", "acd3b9117f3e494f95db213cdccaf293", "5007a0c31beb412c829ce92d48aea596", "c0c7fb0c87774c9d8824bcc9223b2a44", "b489394806cf4fcb9f8ba7ec12876e91", "860cd304c65f44ad84fc0930ba7b97e6", "0ce6125c32cf47ab9c16bf4c3dff8625", "0ea616c1f91d451197f03c82de70f59e", "0598ffdb121742d6bbcfbb06ed1be893", "986cb67f36a44c59824b173b3a94fd99", "fc3ec4c5789643878cd058acb650d131", "ec2b85235e45481693645274c25cf5f6", "7ff41e02023d4d0889f4540944555d4b", "278ae7cd17bf480d8e23e1f325a53625", "b7924f725d5545a48684a10ee54f2ebd", "29525febccb34dbb9ee931e85616650f", "b57071b487004e55a10f4d55c6c20842", "6e084eab76d945f48aef3a7c87fe0423", "1d4457fffddc42a4b7d9565a4a35c67e", "52929dd55eef43488a2c80de407eb309"]} outputId="c3440ba1-e0bb-4477-dbfb-34ce471c8792"
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration 
import huggingface_hub
repo = huggingface_hub.Repository('local_model', clone_from=starting_model_path)
try:
    tokenizer = T5Tokenizer.from_pretrained('local_model')
except:
    tokenizer = T5Tokenizer.from_pretrained(starting_model_path)
try:
    model = FlaxT5ForConditionalGeneration.from_pretrained('local_model')
except:
    model = FlaxT5ForConditionalGeneration.from_pretrained(starting_model_path)

# + id="mZqc_JNhixjS"
# before data is generated we can import libraries to generate it from
import jax, jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.training.common_utils
import flax.training.train_state
import tqdm
import time
import os
# ...
import transformers
import deepspeed
import scipy
import matplotlib
import sentencepiece
import dis

# + id="QnyDTDt_f1fE"
import find_pycode
print('getting training data ...')
tokenizerpfx = starting_model_path.replace('/','_') + '.'
if not train_tokenizer:
  import bytepreservingsentencepiece
  tokenizer.sp_model = bytepreservingsentencepiece.SentencePieceProcessor(tokenizer.vocab_file, **tokenizer.sp_model_kwargs)
find_pycode.write_files('example.', tokenizerpfx, 512, tokenizer, 512, globals(), skip_if_exists = True, train_tokenizer = train_tokenizer, tokenize_binary = True)
if train_tokenizer:
  tokenizer.save_pretrained('local_model')
# repo.push_to_hub(commit_message=f'commit-message', blocking=False)
train_data = find_pycode.read_files('example.', tokenizerpfx, 512, 512, tokenize_binary = True)

# + id="6qTNv8oZwbGS"
#from tokenizers import ByteLevelBPETokenizer
#tokenizer = ByteLevelBPETokenizer()
#tokenizer.train_from_iterator((str for bytes, str in data_tuples), vocab_size=model.config.vocab_size, min_frequency=2) 

# + id="CxCgcJ0dzQY_"


jax.config.update('jax_log_compiles', True)
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache('local_model/cc', max_cache_size_bytes=32*2**30)


# + id="sdJ1Ek3-_j37"
#cmd_args = None
#model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                     model=model,
#                                                     model_parameters=params)



def do_cache():
    # + id="nTQyR1KO4Sbv"
    num_train_steps = len(train_data['input_ids']) // train_batch_size * num_epochs
    
    rng = jax.random.PRNGKey(training_seed)

    # from run_t5_mlm_flax.py
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    
    # + id="m58ESSevKp6P"
    # these are not t5 parameters?
    linear_decay_lr_schedule_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps)
    adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn, b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)
    model = FlaxT5ForConditionalGeneration.from_pretrained('local_model')
    state = flax.training.train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)
    
    # + id="gD_cvz5MebpC"
    def batch_from_indices(dataset : dict, indices):
      #print(dataset['input_ids'].shape, indices.shape)
      result = {
          key : jnp.stack(data[indices,:])
          for key, data in dataset.items()
      }
      # this change could be already put in the dataset passed by the function that produces it
      result['labels'] = result['decoder_input_ids']
      result['decoder_input_ids'] = np.asarray(transformers.models.t5.modeling_flax_t5.shift_tokens_right(result['labels'], tokenizer.pad_token_id, model.config.decoder_start_token_id))
      return result
    
    
    
    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):#input_ids, attention_mask, labels, decoder_input_ids, decoder_attention_mask, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        def loss_fn(params):
            labels = batch.pop('labels')
    
            #logits = state.apply_fn(
            #    input_ids = input_ids,
            #    attention_mask = attention_mask,
            #    decoder_input_ids = decoder_input_ids,
            #    decoder_attention_mask = decoder_attention_mask,
            #    params = params,
            #    dropout_rng = dropout_rng,
            #    train = True
            #).logits
            logits = state.apply_fn(**batch, params = params, dropout_rng = dropout_rng, train = True).logits
            #print(logits.shape)
            #assert len(logits[-1]) == tokenizer.vocab_size
            #logits = logits[0]
    
            # logits, labels, padding_mask=batch['decoder_attention_mask', label_smoothing_factor=0]
            # compute loss
            loss = optax.softmax_cross_entropy(logits, flax.training.common_utils.onehot(labels, logits.shape[-1]))
            padding_mask = batch['decoder_attention_mask']
            loss = (loss * padding_mask).sum() / padding_mask.sum()
    
            #loss = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True).loss
    
            return loss
    
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)
    
        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )
    
        return new_state, metrics, new_dropout_rng
    
    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,), backend=backend)
    
    # Replicate the train state on each device
    state = flax.jax_utils.replicate(state)
    
    print('Performing initial batch to compile train step ...')
    rng, input_rng = jax.random.split(rng)
    num_train_samples = len(train_data['input_ids'])
    train_samples_idx = jax.random.permutation(input_rng, jnp.arange(num_train_samples))
    model_inputs = batch_from_indices(train_data, train_samples_idx[:train_batch_size])
    model_inputs = flax.training.common_utils.shard(model_inputs)
    state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rng=dropout_rngs)
    train_metric = flax.jax_utils.unreplicate(train_metric)
    print('Done.  First loss was', train_metric['loss'].mean())
    

if __name__ == '__main__':
    for train_batch_size in 8*(jax.numpy.arange(64)+1):
        train_batch_size = int(train_batch_size)
        per_device_batch_size = train_batch_size // jax.device_count()
        do_cache()

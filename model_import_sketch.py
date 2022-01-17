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
# <a href="https://colab.research.google.com/github/xloem/techsketball/blob/wip/model_import_sketch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + colab={"base_uri": "https://localhost:8080/"} id="nH1Ld_vd9wyx" outputId="c40a7044-01cc-4cc2-935f-8fdfc5793548"
#[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)

import jax

starting_model_path = 't5-base'#'t5-small'#'bigscience/T0pp'
input_width = 512
# these are not t5 parameters?
train_batch_size = 20 # small for notebook
per_device_batch_size = train_batch_size // jax.device_count()
num_epochs = 10
training_seed = 0
learning_rate = 3e-4


# #!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# !pip3 install deepspeed
# !pip3 install transformers
# !pip3 install flax
# !pip3 install sentencepiece
# !git clone https://github.com/xloem/techsketball && ln -s techsketball/* .

# + colab={"base_uri": "https://localhost:8080/"} id="mTGBjrXoX2eS" outputId="4a7b7590-2846-4ac4-822d-a9a540cbafa8"
import jax.tools.colab_tpu
import jaxlib
import os
try:
  if 'COLAB_TPU_ADDR' in os.environ:
    jax.tools.colab_tpu.setup_tpu()
  jaxlib.xla_extension.tpu_client()
  backend = 'tpu'
except:
  try:
    jaxlib.xla_extension.gpu_client()
    backend = 'gpu'
  except:
    backend = 'cpu'
import tensorflow as tf
# !nvidia-smi
jax.local_devices()

# + id="qhJhFJfQAOXG"
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration 
import huggingface_hub
#repo = huggingface_hub.Repository('local_model', clone_from='hub_model_id')
tokenizer = T5Tokenizer.from_pretrained(starting_model_path) # only for source, not for binary
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
import scipy

# + id="QnyDTDt_f1fE" outputId="6252ccb7-ed2a-4db8-edcb-0ac73bb3dfad" colab={"base_uri": "https://localhost:8080/"}
import find_pycode
print('getting training data ...')
tokenizerpfx = starting_model_path.replace('/','_') + '.'
find_pycode.write_files('example.', tokenizerpfx, 512, tokenizer, 512, globals(), skip_if_exists = True, tokenize_binary = True)
tokenizer.save_pretrained('local_model')
train_data = find_pycode.read_files('example.', tokenizerpfx, 512, 512)

# + id="6qTNv8oZwbGS"
#from tokenizers import ByteLevelBPETokenizer
#tokenizer = ByteLevelBPETokenizer()
#tokenizer.train_from_iterator((str for bytes, str in data_tuples), vocab_size=model.config.vocab_size, min_frequency=2) 

# + id="CxCgcJ0dzQY_"



# + id="sdJ1Ek3-_j37"
#import deepspeed
#cmd_args = None
#model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                     model=model,
#                                                     model_parameters=params)

# + id="nTQyR1KO4Sbv"
num_train_steps = len(train_data['input_ids']) // train_batch_size * num_epochs

rng = jax.random.PRNGKey(training_seed)


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


# + id="m58ESSevKp6P" outputId="19801cbf-211d-43f5-e143-3b1ad233e1d5" colab={"base_uri": "https://localhost:8080/"}
# these are not t5 parameters?
linear_decay_lr_schedule_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps)
adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn, b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)
state = flax.training.train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

jax.config.update('jax_log_compiles', True)


# from run_t5_mlm_flax.py
dropout_rngs = jax.random.split(rng, jax.local_device_count())

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
p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0,), backend=backend)

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


# + id="doKKw-W345Zn" outputId="dd3510d2-b9b7-485a-8ef9-6ccad799bed9" colab={"base_uri": "https://localhost:8080/"}


train_time = 0
epochs = tqdm.tqdm(range(num_epochs), desc="Epoch ... ", position=0)
for epoch in epochs:
    # ======================== Training ================================
    train_start = time.time()
    train_metrics = []

    # Create sampling rng
    rng, input_rng = jax.random.split(rng)

    # Generate an epoch by shuffling sampling indices from the train dataset
    num_train_samples = len(train_data['input_ids'])
    train_samples_idx = jax.random.permutation(input_rng, jnp.arange(num_train_samples))
    #train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)

    # Gather the indexes for creating the batch and do a training step
    for step, batch_idx in enumerate(tqdm.tqdm(range(num_train_samples // train_batch_size), desc="Training...", position=1)):
        #samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx]
        model_inputs = batch_from_indices(train_data, train_samples_idx[train_batch_size * batch_idx:train_batch_size * batch_idx + train_batch_size])
        #print('model_inputs are', {key:val.shape for key, val in model_inputs.items()})
        #model_inputs = data_collator(samples)

        # Model forward
        model_inputs = flax.training.common_utils.shard(model_inputs)
        state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rng=dropout_rngs)
        train_metrics.append(train_metric)

        cur_step = epoch * (num_train_samples // train_batch_size) + step

        if cur_step % training_args.logging_steps == 0 and cur_step > 0 and jax.process_index() == 0:
            # Save metrics
            train_metric = flax.jax_utils.unreplicate(train_metric)
            train_time += time.time() - train_start
            #if has_tensorboard and jax.process_index() == 0:
            #    write_train_metric(summary_writer, train_metrics, train_time, cur_step)

            epochs.write(
                f"Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()}"
            )

            train_metrics = []
        if cur_step % (num_train_samples // 8) == 0:
            # save checkpoint
            if jax.process_index() == 0:
                params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                model.save_pretrained('local_model', params=params)
                # repo.push_to_hub(commit_message=f'commit-message', blocking=False)

# + id="aGdYRKVJxHxN"
while True:
  print(repr(eval(input('>>> '), globals(), locals())))

# + [markdown] id="K2lLdDyZP0mT"
#

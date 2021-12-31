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

# + id="qhJhFJfQAOXG"
from transformers import T5Tokenizer, FlaxT5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained(starting_model_path) # only for source, not for binary
#model = FlaxT5ForConditionalGeneration.from_pretrained(starting_model_path)

# + colab={"base_uri": "https://localhost:8080/"} id="QnyDTDt_f1fE" outputId="df6a10a9-d449-411d-87ce-e779ef83222c"
from find_pycode import pair_finder
print('generating training data ...')
data_tuples = [*pair_finder(globals())] # finds code examples

# + id="TrRadfeAhZSd"
import jax, numpy as np
data_input_ids = jax.numpy.stack([
  np.frombuffer(bytes.ljust(input_width, b'\0'), dtype=np.uint8)
  for bytes, str in data_tuples
  if len(bytes) <= input_width
])
data_labels = [str for bytes, str in data_tuples if len(bytes) <= input_width]


# + id="6qTNv8oZwbGS"
#from tokenizers import ByteLevelBPETokenizer
#tokenizer = ByteLevelBPETokenizer()
#tokenizer.train_from_iterator((str for bytes, str in data_tuples), vocab_size=model.config.vocab_size, min_frequency=2) 

# + id="CxCgcJ0dzQY_"
data_labels_tokenized = tokenizer(data_labels, padding = True, return_tensors = 'np')
data_label_ids = data_labels_tokenized['input_ids']
data_label_attention_masks = data_labels_tokenized['attention_mask']
#data_label_ids[data_label_attention_masks == 0] = -100

# + id="sdJ1Ek3-_j37"
#import deepspeed
#cmd_args = None
#model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
#                                                     model=model,
#                                                     model_parameters=params)

# + id="nTQyR1KO4Sbv"
# these are not t5 parameters
per_device_batch_size = 16 # small for notebook
num_epochs = 10
training_seed = 0
learning_rate = 3e-4

total_batch_size = per_device_batch_size * jax.device_count()
num_train_steps = len(data_input_ids) // total_batch_size * num_epochs

# + id="doKKw-W345Zn"
import optax, flax.training.train_state
linear_decay_lr_schedule_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps)
adamw = optax.adamw(learning_rate=linear_decay_lr_schedule_fn, b1=0.9, b2=0.98, eps=1e-8, weight_decay=0.01)
state = flax.training.train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

def data_loader(rng, dataset, batch_size, shuffle=True):
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch

# from run_t5_mlm_flax.py
#rng = jax.random.PRNGKey(training_seed)
#dropout_rngs = jax.random.split(rng, jax.local_device_count())
#
## Define gradient update step fn
#def train_step(state, batch, dropout_rng):
#    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
#                                                                                     def loss_fn(params):
#        labels = batch.pop("labels")
#
#        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_r
#ngain=True)[0]
#                                                                                         # compute loss
#        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])).mean()
#
#        return loss
#    grad_fn = jax.value_and_grad(loss_fn)
#    loss, grad = grad_fn(state.params)
#    grad = jax.lax.pmean(grad, "batch")
#    new_state = state.apply_gradients(grads=grad)
#    metrics = jax.lax.pmean(
#        {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.ste
#p)xis_name="batch"
#    )
#    return new_state, metrics, new_dropout_rng
#
## Create parallel version of the train step
#p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
## Replicate the train state on each device
#state = jax_utils.replicate(state)
#
#train_time = 0
#epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
#for epoch in epochs:                                                                 # ======================== Training ================================             train_start = time.time()                                                        train_metrics = []                                                                                                                                                # Create sampling rng                                                            rng, input_rng = jax.random.split(rng)
#                                                                                         # Generate an epoch by shuffling sampling indices from the train dataset
#        num_train_samples = len(tokenized_datasets["train"])
#        train_samples_idx = jax.random.permutation(input_rng, jnp.arange(num_train_samples))                                                                              train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)
#        # Gather the indexes for creating the batch and do a training step               for step, batch_idx in enumerate(tqdm(train_batch_idx, desc="Training...", position=1)):
#            samples = [tokenized_datasets["train"][int(idx)] for idx in batch_idx
#]                                                                                            model_inputs = data_collator(samples)                                                                                                                             # Model forward
#            model_inputs = shard(model_inputs.data)                                          state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)                                                                               train_metrics.append(train_metric)
#                                                                                             cur_step = epoch * (num_train_samples // train_batch_size) + step
#                                                                                             if cur_step % training_args.logging_steps == 0 and cur_step > 0:                     # Save metrics                                                                   train_metric = jax_utils.unreplicate(train_metric)                               train_time += time.time() - train_start                                          if has_tensorboard and jax.process_index() == 0:                                     write_train_metric(summary_writer, train_metrics, train_time, cur_step)                                                                                                                                                                        epochs.write(                                                                        f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"                                          )                                                                                                                                                                 train_metrics = []
#

# + id="aGdYRKVJxHxN"
while True:
  print(repr(eval(input('>>> '), globals(), locals())))

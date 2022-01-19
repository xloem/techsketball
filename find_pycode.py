# finds pairs of source and bytecode within the current python runtime, to generate data
# can generate and load sets of 4 files with these data prepped for a translation model
# generates and loads example files when run (at bottom)

import collections
import types

import inspect # can get source code
import marshal # can convert functions to their filesystem bytecode


# a generator yielding matching pairs of bytecode and source found by tree exploration from passed objects
class pair_finder:
    def __init__(self, *initial_objects):
        self.found = set()
        self.queue = collections.deque()
        self._addnew(*(((), obj) for obj in initial_objects))
    def _addnew(self, *item_tuples):
        for hist, item in item_tuples:
            ident = id(item)
            if ident not in self.found:
                self.found.add(ident)
                self.queue.append((hist,item))
    def __iter__(self):
        while self.queue:
            hist, item = self.queue.popleft()
            try:
                if isinstance(item, types.FunctionType):
                    try:
                        src = inspect.getsource(item)
                        bin = marshal.dumps(item.__code__)
                        yield bin, src
                    except:
                        pass
                elif type(item) in (dict, types.MappingProxyType):
                    self._addnew(*(((*hist, name), key) for name, key in item.items()))
                elif hasattr(item, '__dict__'):
                    self._addnew((hist, item.__dict__))
                elif hasattr(item, '__class__'):
                    self._addnew((hist, item.__class__))
                elif type(item) in (str, types.BuiltinFunctionType, types.BuiltinMethodType) or item in (None,) or hist[-1].startswith('__'):
                    continue
                else:
                    print(hist, type(item))
                    import pdb; pdb.set_trace()
                    assert not 'hist, item not recognised type'
            except ReferenceError:
                continue
    def train_tokenizer(self, tokenizer, pfx, tokenizerpfx, skip_if_exists = False, verbose = True, vocab_size = None):
        import bytepreservingsentencepiece as spm
        vocab_size = vocab_size if vocab_size is not None else tokenizer.sp_model.vocab_size()
        filename = self.train_bytespm(pfx, tokenizerpfx, vocab_size, skip_if_exists = skip_if_exists, verbose = verbose, unk_id = tokenizer.unk_token_id, bos_id = tokenizer.bos_token_id, eos_id = tokenizer.eos_token_id, pad_id = tokenizer.pad_token_id)
        tokenizer.vocab_file = filename
        tokenizer.sp_model = spm.BytePreservingSentencePieceProcessor(**tokenizer.sp_model_kwargs, model_file=tokenizer.vocab_file)
        return tokenizer
    def train_bytespm(self, pfx, tokenizerpfx, vocab_size, skip_if_exists = False, verbose = True, unk_id = 0, bos_id = 1, eos_id = 2, pad_id = -1):
        unk_id = unk_id if unk_id is not None else -1
        bos_id = bos_id if bos_id is not None else -1
        eos_id = eos_id if eos_id is not None else -1
        pad_id = pad_id if pad_id is not None else -1
        filename = f'{pfx}{tokenizerpfx}{vocab_size}.spm'
        # tokenizer.vocab_file = filename
        # pair_finder().train_spm(tokenizer.vocab_file, tokenizer.sp_model.vocab_size())
        # tokenizer.sp_model.Load(tokenizer.vocab_file)
        import bytepreservingsentencepiece as spm
        import os
        if skip_if_exists:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                return filename
        with open(filename, 'wb') as spm_file:
            spm.SentencePieceTrainer.train(sentence_iterator = (src for bin, src in self), model_writer = spm_file, vocab_size = vocab_size, character_coverage = 1.0, model_type = 'unigram', max_sentence_length = 65536, minloglevel = 0 if verbose else 1, unk_id = unk_id, bos_id = bos_id, eos_id = eos_id, pad_id = pad_id)
        return filename

# takes some strings and model parameters and generates tensor files for passed objects using above class
def write_files(pfx, tokenizerpfx, input_width, tokenizer, label_width, *initial_objects, verbose = True, skip_if_exists = False, tokenize_binary = False, tokenize_source = True, train_tokenizer = False, vocab_size = None, excluded_strings = ['@']):
    import numpy as np
    if train_tokenizer:
        tokenizer = pair_finder(*initial_objects).train_tokenizer(tokenizer, pfx, tokenizerpfx, skip_if_exists = skip_if_exists, vocab_size = vocab_size)
    if skip_if_exists:
        import os
        if os.path.exists(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec') and os.path.getsize(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec') > 0:
            return
    if tokenize_binary:
        binpfx = tokenizerpfx
    else:
        binpfx = ''
    with open(f'{pfx}{binpfx}bin.u16.{input_width}vec', 'wb') as itok, open(f'{pfx}{binpfx}bin.attnmask.u8.{input_width}vec', 'wb') as iattn, open(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec', 'wb') as otok, open(f'{pfx}{tokenizerpfx}src.attnmask.u8.{label_width}vec', 'wb') as oattn:
        iattn_buf = np.zeros(input_width, dtype=np.uint8)
        for idx, (bin, src) in enumerate(pair_finder(*initial_objects)):
            if any((excluded_string in src for excluded_string in excluded_strings)):
                continue
            if tokenize_binary:
                bin_tokenized = tokenizer(bin.decode('iso-8859-1'), padding = 'max_length', return_tensors = 'np', max_length = input_width)
                iattn_buf[:] = bin_tokenized['attention_mask'][0][:len(iattn_buf)]
                # this assertion doesn't presently pass
                # notably spaces with a following quote have a space lopped off
                #assert tokenizer.decode(bin_tokenized['input_ids'][0], skip_special_tokens = True).encode('iso-8859-1') == bin
                bin = bin_tokenized['input_ids'][0]
            else:
                bin = np.frombuffer(bin.ljust(input_width, b'\0'), dtype=np.uint8)
                iattn_buf[:len(bin)] = 1
                iattn_buf[len(bin):] = 0
            if len(bin) > input_width:
                continue
            if tokenize_source:
                srctok = tokenizer(src, padding = 'max_length', return_tensors = 'np', max_length = label_width)
            else:
                srctok = np.frombuffer(src.encode('iso-8859-1'), dtype=np.uint8)
            if len(srctok['input_ids'][0]) > label_width:
                continue
            itok.write(bin.astype(np.uint16).data)
            iattn.write(iattn_buf.data)
            len1 = otok.write(srctok['input_ids'].astype(np.uint16).data)
            len2 = oattn.write(srctok['attention_mask'].astype(np.uint8).data)
            if len1 != label_width * 2 or len2 != label_width:
                import pdb; pdb.set_trace()
            if verbose and not (idx & 0xf):
                print(str(idx), 'generated', end='\r', flush = True)
    if verbose:
        print()

# loads tensor files using instant memmap and returns them as dict usable as T5 model kwparams
def read_files(pfx, tokenizerpfx, input_width, label_width, tokenize_binary = False):
    import numpy as np
    import mmap
    import os
    if tokenize_binary:
        binpfx = tokenizerpfx
    else:
        binpfx = ''
    with open(f'{pfx}{binpfx}bin.u16.{input_width}vec', 'rb') as itok, open(f'{pfx}{binpfx}bin.attnmask.u8.{input_width}vec', 'rb') as iattn, open(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec', 'rb') as otok, open(f'{pfx}{tokenizerpfx}src.attnmask.u8.{label_width}vec', 'rb') as oattn:
        itok = np.frombuffer(mmap.mmap(itok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint16)
        iattn = np.frombuffer(mmap.mmap(iattn.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint8)
        otok = np.frombuffer(mmap.mmap(otok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint16)
        oattn = np.frombuffer(mmap.mmap(oattn.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint8)
    return dict(
        input_ids = itok.reshape(-1, input_width),
        attention_mask = iattn.reshape(-1, input_width),
        decoder_input_ids = otok.reshape(-1, label_width),
        decoder_attention_mask = oattn.reshape(-1, label_width)
    )


if __name__ == '__main__':
    # generates some small files using t5-small's tokenizer, and prints a random byte/src pair from them
    import numpy as np
    import transformers, sentencepiece as spm
    
    modelname = 't5-small'
    tokenizer = transformers.T5Tokenizer.from_pretrained(modelname)
    tokenizerpfx = modelname.replace('/','_') + '.train.'
    
    write_files('example.', tokenizerpfx, 512, tokenizer, 512, globals(), train_tokenizer = True, skip_if_exists = True, tokenize_binary = True, vocab_size = 31000)
    print('Wrote 4 example.* files.')
    
    result = read_files('example.', tokenizerpfx, 512, 512, tokenize_binary = True)
    
    example_idx = np.random.randint(len(result['input_ids']))
    print(f"Here's pair #{example_idx}:")
    
    bytecode = tokenizer.decode(
            result['input_ids'][example_idx][result['attention_mask'][example_idx] != 0]
    ).encode('iso-8859-1')
    src = tokenizer.decode(
            result['decoder_input_ids'][example_idx][result['decoder_attention_mask'][example_idx] != 0]
    )
    print('Bytes =', bytecode)
    print('Source:')
    print(src)

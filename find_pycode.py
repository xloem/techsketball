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

# takes some strings and model parameters and generates tensor files for passed objects using above class
def write_files(pfx, tokenizerpfx, input_width, tokenizer, label_width, *initial_objects, verbose = True, skip_if_exists = False):
    import numpy as np
    if skip_if_exists:
        import os
        if os.path.exists(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec'):
            return
    with open(f'{pfx}bin.u8.{input_width}vec', 'wb') as itok, open(f'{pfx}bin.attnmask.u8.{input_width}vec', 'wb') as iattn, open(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec', 'wb') as otok, open(f'{pfx}{tokenizerpfx}src.attnmask.u8.{label_width}vec', 'wb') as oattn:
        iattn_buf = np.zeros(input_width, dtype=np.uint8)
        for idx, (bin, src) in enumerate(pair_finder(globals())):
            if len(bin) > input_width:
                continue
            srctok = tokenizer(src, padding = 'max_length', return_tensors = 'np')
            if len(srctok['input_ids'][0]) > label_width:
                continue
            itok.write(bin.ljust(input_width, b'\0'))
            iattn_buf[:len(bin)] = 1
            iattn_buf[len(bin):] = 0
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
def read_files(pfx, tokenizerpfx, input_width, label_width):
    import numpy as np
    import mmap
    with open(f'{pfx}bin.u8.{input_width}vec', 'rb') as itok, open(f'{pfx}bin.attnmask.u8.{input_width}vec', 'rb') as iattn, open(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec', 'rb') as otok, open(f'{pfx}{tokenizerpfx}src.attnmask.u8.{label_width}vec', 'rb') as oattn:
        itok = np.frombuffer(mmap.mmap(itok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint8)
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
    from transformers import T5Tokenizer
    
    modelname = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(modelname)
    tokenizerpfx = modelname.replace('/','_') + '.'
    
    write_files('example.', tokenizerpfx, 512, tokenizer, 512, globals(), skip_if_exists = True)
    print('Wrote 4 example.* files.')
    
    result = read_files('', tokenizerpfx, 512, 512)
    
    example_idx = np.random.randint(len(result['input_ids']))
    print(f"Here's pair #{example_idx}:")
    
    bytecode = result['input_ids'][example_idx][result['attention_mask'][example_idx] != 0].tobytes()
    src = tokenizer.decode(
            result['decoder_input_ids'][example_idx][result['decoder_attention_mask'][example_idx] != 0]
        )
    print('Bytes =', bytecode)
    print('Source:')
    print(src)

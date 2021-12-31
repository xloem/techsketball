# finds pairs of source and bytecode within the current python runtime, to generate data

import collections
import types

import inspect # can get source code
import marshal # can convert functions to their filesystem bytecode

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

def write_files(pfx, tokenizerpfx, input_width, tokenizer, label_width, *initial_objects, verbose = True):
    import numpy as np
    with open('{pfx}bin.u8.{input_width}vec', 'wb') as itok, open('{pfx}bin.attnmask.u8.{input_width}vec', 'wb') as iattn, open(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec', 'wb') as otok, open(f'{pfx}{tokenizerpfx}.src.attnmask.u8.{label_width}vec', 'wb') as oattn:
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

def read_files(pfx, tokenizerpfx, input_width, label_width):
    import numpy as np
    import mmap
    with open('{pfx}bin.u8.{input_width}vec', 'rb') as itok, open('{pfx}bin.attnmask.u8.{input_width}vec', 'rb') as iattn, open(f'{pfx}{tokenizerpfx}src.u16.{label_width}vec', 'rb') as otok, open(f'{pfx}{tokenizerpfx}.src.attnmask.u8.{label_width}vec', 'rb') as oattn:
        itok = np.frombuffer(mmap.mmap(itok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint8)
        iattn = np.frombuffer(mmap.mmap(itok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint8)
        otok = np.frombuffer(mmap.mmap(itok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint16)
        oattn = np.frombuffer(mmap.mmap(itok.fileno(), 0, access = mmap.ACCESS_READ, offset = 0), np.uint16)
    return dict(
        input_ids = itok,
        attention_mask = iattn,
        decoder_input_ids = otok,
        decoder_attention_mask = oattn
    )

if __name__ == '__main__':
    from transformers import T5Tokenizer
    model = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model)
    tokenizerpfx = model.replace('/','_') + '.'
    write_files('', tokenizerpfx, 512, tokenizer, 512, globals())
    result = read_files('', tokenizerpfx, 512, 512)
    print(result['input_ids'][1].tobytes(), tokenizer.decode(result['decoder_input_ids'][1]))

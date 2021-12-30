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
if __name__ == '__main__':
    for bin, src in pair_finder(globals()):
        print(repr(bin), repr(src))

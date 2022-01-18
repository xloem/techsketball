import transformers

import bytepreservingsentencepiece as spm

class Model:
    def __init__(self, model = 'baffo32/pyc2py_alpha'):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(model)
        self.model = transformers.FlaxT5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer.sp_model = spm.SentencePieceProcessor(self.tokenizer.vocab_file, **self.tokenizer.sp_model_kwargs)
    def guess(self, bytes):
        text = bytes.decode('iso-8859-1')
        tokenized_input = self.tokenizer.encode(text, return_tensors='jax')
        output = self.model.generate(tokenized_input, pad_token_id=self.tokenizer.pad_token_id, do_sample=True)
        return self.tokenizer.decode(output['sequences'][0], skip_special_tokens=True)
    @staticmethod
    def bytecode(func):
        import marshal
        return marshal.dumps(func.__code__)
    @staticmethod
    def set_default_backend(name = None):
        import jax
        if name is not None:
            jax._src.lib.xla_bridge._default_backend = jax._src.lib.xla_bridge.backends()[name]

if __name__ == '__main__':
    Model.set_default_backend()#'cpu')#'gpu')#'tpu')
    model = Model()

    def example_sum(a, b):
        return a + b

    import inspect
    print(inspect.getsource(example_sum))

    bytecode = model.bytecode(example_sum)
    print(bytecode)

    src = model.guess(bytecode)
    print(src)

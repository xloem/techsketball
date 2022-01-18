import transformers

import bytepreservingsentencepiece as spm

class Model:
    def __init__(self, model = 'baffo32/pyc2py_alpha'):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(model)
        self.model = transformers.FlaxT5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer.sp_model = spm.SentencePieceProcessor(tokenizer.vocab_file, **tokenizer.sp_kwargs)
    def guess(self, bytes):
        tokenized_input = self.tokenizer.encode(bytes, return_tensors='jax')
        output = self.model.generate(tokenized_input, pad_token_id=self.tokenizer.pad_token_id, do_sample=True)
        return self.tokenizer.decode(output, skip_special_tokens=True)
    @staticmethod
    def bytecode(self, func):
        import marshal
        return marshal.dumps(func.__code__)

if __name__ == '__main__':
    def sum(a, b):
        return a + b
    import inspect
    print(inspect.getsource(sum))
    bytecode = Model.bytecode(sum)
    print(bytecode)
    src = Model().guess(bytecode)
    print(src)

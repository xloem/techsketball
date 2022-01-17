import sentencepiece as spm
import numpy as np

#NUL_PIECE = chr(256) + 'NUL'

class BytePreservingSentencePieceTrainer:
    @staticmethod
    def Train(sentence_iterator = None, vocab_size = None, **kwparams):
        #def sentence_generator():
        #    #import pdb; pdb.set_trace()
        #    #for byte in range(1, 257): # sentencepiece doesn't support 0 chars, so 0 is passed as 256
        #    #    yield chr(byte)
        #    yield from sentence_iterator
        #for piece_id in 'unk_id bos_id eos_id'.split(' '):
        #    if piece_id in kwparams and kwparams[piece_id] >= 0:
        #        kwparams[piece_id] += 256
        #return spm.SentencePieceTrainer.Train(sentence_iterator = sentence_generator(), vocab_size = vocab_size - 256, user_defined_symbols = NUL_PIECE, **kwparams)
        return spm.SentencePieceTrainer.Train(sentence_iterator = sentence_generator(), vocab_size = vocab_size - 256, **kwparams)
    train = Train
SentencePieceTrainer = BytePreservingSentencePieceTrainer

class BytePreservingSentencePieceProcessor(spm.SentencePieceProcessor):
    def __init__(self, *params, **kwparams):
        spm.SentencePieceProcessor.__init__(self, *params, **kwparams)
        ##self.byte_to_inner_id = [ spm.SentencePieceProcessor.piece_to_id(self, chr(x)) for x in range(256) ]
        ##self.inner_id_to_byte = { self.byte_to_inner_id[x] : x for x in range(256) }
        #self.ord_to_piece = []
        #for ord_id in range(256):
        #    pieces = spm.SentencePieceProcessor.encode(self, chr(ord_id), out_type=str)
        #    if self._add_bos:
        #        pieces = pieces[1:]
        #    if self._add_eos:
        #        pieces = pieces[:-1]
        #    if len(pieces) == 1:
        #        self.ord_to
        idx = -2 if self._add_eos else -1
        #self.ord_to_piece = [
        #        spm.SentencePieceProcessor.encode(self, '\x00' + chr(x), out_type=str)[idx]
        #        for x in range(256)
        #]
        self.ord_to_inner_id = np.array([
                spm.SentencePieceProcessor.encode(self, '\x00' + chr(x))[idx]
                for x in range(256)
        ])
        #self.piece_to_ord = {
        #    self.ord_to_piece[x] : x
        #    for x in range(256)
        #}
        self.inner_id_to_outer_id = np.arange(self.vocab_size()) + 256
        for ord, inner_id in enumerate(self.ord_to_inner_id):
            self.inner_id_to_outer_id[inner_id] = ord

    def PieceToId(self, str):
        ##if str == NUL_PIECE:
        ##    return 0
        #try:
        #    num = ord(str)
        #except:
        #    num = 256
        #if num < 256:
        #    return num
        #ord = self.piece_to_ord.get(str)
        #if ord is not None:
        #    return ord
        #else:
        #   return spm.SentencePieceProcessor.piece_to_id(self, str) + 256
        #import pdb; pdb.set_trace()
        return self.inner_id_to_outer_id[super().piece_to_id(str)].item()
    piece_to_id = PieceToId

    def IdToPiece(self, id):
        #if id == 0:
        #    return NUL_PIECE
        if id < 256:
            return chr(id)
        else:
            return spm.SentencePieceProcessor.id_to_piece(self, id - 256)
    id_to_piece = IdToPiece

    def Encode(self, input, out_type=None, **kwparams):
        if out_type is not str:
            raise NotImplementedError
        #if type(input) is str:
        #    assert NUL_PIECE not in input # this can probably be commented out in an emergency
        #                                  # NUL_PIECE can also be escaped, or the tokenizer
        #                                  # can be retrained after it is changed
        #    input = input.replace('\0', NUL_PIECE)
        return spm.SentencePieceProcessor.Encode(self, input, out_type=out_type, **kwparams)
    encode = Encode

    decode = NotImplemented
    Decode = NotImplemented

    def GetPieceSize(self):
        return spm.SentencePieceProcessor.GetPieceSize(self) + 256
    get_piece_size = GetPieceSize
    vocab_size = GetPieceSize

    def bos_id(self):
        return spm.SentencePieceProcessor.bos_id(self) + 256
    def eos_id(self):
        return spm.SentencePieceProcessor.eos_id(self) + 256
    def pad_id(self):
        return spm.SentencePieceProcessor.pad_id(self) + 256
    def unk_id(self):
        return spm.SentencePieceProcessor.unk_id(self) + 256

SentencePieceProcessor = BytePreservingSentencePieceProcessor

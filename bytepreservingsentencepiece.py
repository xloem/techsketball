import sentencepiece as spm
import numpy as np

class BytePreservingSentencePieceTrainer:
    @staticmethod
    def Train(sentence_iterator = None, vocab_size = None, **kwparams):
        return spm.SentencePieceTrainer.Train(sentence_iterator = sentence_generator(), vocab_size = vocab_size - 256, **kwparams)
    train = Train
SentencePieceTrainer = BytePreservingSentencePieceTrainer

class BytePreservingSentencePieceProcessor(spm.SentencePieceProcessor):
    def __init__(self, *params, **kwparams):
        spm.SentencePieceProcessor.__init__(self, *params, **kwparams)
        idx = -2 if self._add_eos else -1
        self.ord_to_inner_id = np.array([
                spm.SentencePieceProcessor.encode(self, '\x00' + chr(x))[idx]
                for x in range(256)
        ])
        self.inner_id_to_outer_id = np.arange(self.vocab_size()) + 256
        for ord, inner_id in enumerate(self.ord_to_inner_id):
            self.inner_id_to_outer_id[inner_id] = ord

    def PieceToId(self, str):
        return self.inner_id_to_outer_id[super().piece_to_id(str)].item()
    piece_to_id = PieceToId

    def IdToPiece(self, id):
        if id < 256:
            return chr(id)
        else:
            return spm.SentencePieceProcessor.id_to_piece(self, id - 256)
    id_to_piece = IdToPiece

    def Encode(self, input, out_type=None, **kwparams):
        if out_type is not str:
            raise NotImplementedError
        return spm.SentencePieceProcessor.Encode(self, input, out_type=out_type, **kwparams)
    encode = Encode

    decode = NotImplemented
    Decode = NotImplemented

    def GetPieceSize(self):
        return spm.SentencePieceProcessor.GetPieceSize(self) + 256
    get_piece_size = GetPieceSize
    vocab_size = GetPieceSize

    def bos_id(self):
        bos_id = spm.SentencePieceProcessor.bos_id(self)
        if bos_id >= 0:
            return bos_id + 256
        else:
            return bos_id
    def eos_id(self):
        eos_id = spm.SentencePieceProcessor.eos_id(self)
        if eos_id >= 0:
            return eos_id + 256
        else:
            return eos_id
    def pad_id(self):
        pad_id = spm.SentencePieceProcessor.pad_id(self)
        if pad_id >= 0:
            return pad_id + 256
        else:
            return pad_id
    def unk_id(self):
        unk_id = spm.SentencePieceProcessor.unk_id(self)
        if unk_id >= 0:
            return unk_id + 256
        else:
            return unk_id

SentencePieceProcessor = BytePreservingSentencePieceProcessor

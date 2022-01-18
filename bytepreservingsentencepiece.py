import sentencepiece as spm
import numpy as np

class BytePreservingSentencePieceTrainer:
    @staticmethod
    def Train(sentence_iterator = None, vocab_size = None, **kwparams):
        for sym_id in 'unk_id eos_id bos_id pad_id'.split(' '):
            if sym_id in kwparams and kwparams[sym_id] >= 0 and kwparams[sym_id] < 256:
                kwparams[sym_id] += 256
        def sentence_generator():
            for x in range(256):
                yield chr(x)
            yield from sentence_iterator
        return spm.SentencePieceTrainer.Train(sentence_iterator = sentence_generator(), vocab_size = vocab_size, **kwparams, normalization_rule_name = 'identity', remove_extra_whitespaces = False, byte_fallback = True)
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
        self.inner_id_to_outer_id = np.arange(self.vocab_size())
        available_high_ids = set()
        needs_mapping_queue = [(inner_id, ord) for ord, inner_id in enumerate(self.ord_to_inner_id)]
        already_mapped_ids = set()
        while len(needs_mapping_queue):
            inner_id, destination = needs_mapping_queue.pop()
            if destination is None:
                if len(available_high_ids) == 0:
                    needs_mapping_queue.insert(0, (inner_id, destination))
                    continue
                destination = available_high_ids.pop()
            self.inner_id_to_outer_id[inner_id] = destination
            if inner_id >= 256:
                available_high_ids.add(inner_id)
            already_mapped_ids.add(inner_id)
            if destination not in already_mapped_ids:
                needs_mapping_queue.append((destination, None))

        #for ord, inner_id in enumerate(self.ord_to_inner_id):
        #    if ord != inner_id:
        #        if inner_id >= 256:
        #            self.inner_id_to_outer_id[inner_id] = ord
        #            self.inner_id_to_outer_id[ord] = inner_id
        #        else:


    def PieceToId(self, str):
        return self.inner_id_to_outer_id[super().piece_to_id(str)].item()
    piece_to_id = PieceToId

    def IdToPiece(self, id):
        if id < 256:
            return chr(id)
        else:
            return spm.SentencePieceProcessor.id_to_piece(self, id)
    id_to_piece = IdToPiece

    def Encode(self, input, out_type=None, **kwparams):
        if out_type is not str:
            raise NotImplementedError
        return spm.SentencePieceProcessor.Encode(self, input, out_type=out_type, **kwparams)
    encode = Encode

    decode = NotImplemented
    Decode = NotImplemented

    def GetPieceSize(self):
        return spm.SentencePieceProcessor.GetPieceSize(self)
    get_piece_size = GetPieceSize
    vocab_size = GetPieceSize

    def bos_id(self):
        bos_id = spm.SentencePieceProcessor.bos_id(self)
        return self.inner_id_to_outer_id(bos_id).item()
    def eos_id(self):
        eos_id = spm.SentencePieceProcessor.eos_id(self)
        return self.inner_id_to_outer_id(eos_id).item()
    def pad_id(self):
        pad_id = spm.SentencePieceProcessor.pad_id(self)
        return self.inner_id_to_outer_id(pad_id).item()
    def unk_id(self):
        unk_id = spm.SentencePieceProcessor.unk_id(self)
        return self.inner_id_to_outer_id(unk_id).item()

SentencePieceProcessor = BytePreservingSentencePieceProcessor

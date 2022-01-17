import sentencepiece as spm

# plan:
# - when training, prepend byte-per-sentence to each data
#   this will have it learn to make tokens for individual bytes
#   (trainer)
# - mutate the token ids of the input and output,
#   so that the byte identities for these values are preserved !
#   (processor)

class BytePreservingSentencePieceTrainer:
    @staticmethod
    def Train(sentence_iterator = None, vocab_size = None, **kwparams):
        def sentence_generator():
            #import pdb; pdb.set_trace()
            for byte in range(256):
                # todo: zero chars should be replaced with a special token
                yield chr(byte)
            yield from sentence_iterator
        return spm.SentencePieceTrainer.Train(sentence_iterator = sentence_generator(), vocab_size = vocab_size - 256, **kwparams)
    train = Train
SentencePieceTrainer = BytePreservingSentencePieceTrainer

class BytePreservingSentencePieceProcessor(spm.SentencePieceProcessor):
    def __init__(self, *params, **kwparams):
        spm.SentencePieceProcessor.__init__(self, *params, **kwparams)
        self.byte_to_id = [ spm.SentencePieceProcessor.piece_to_id(self, chr(x)) for x in range(256) ]
        self.id_to_byte = { self.byte_to_id[x] : x for x in range(256) }

    def PieceToId(self, str):
        try:
            num = ord(str)
        except:
            num = 256
        if num < 256:
            return num
        else:
            return spm.SentencePieceProcessor.piece_to_id(self, str) + 256

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
SentencePieceProcessor = BytePreservingSentencePieceProcessor

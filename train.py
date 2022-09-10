import torch
from gensim.models import Word2Vec
from torch import nn
from tqdm import tqdm

from models import Doc2Vec, TFIDFWrapper, Word2VecWrapper
from preprocessing import Preprocessor, preprocess_bible

if __name__ == "__main__":
    preprocessor = Preprocessor()
    word2vec = Word2VecWrapper(vector_size=128, window=5, workers=-1)

    with open("data/bible.txt", "r") as fin:
        text = preprocess_bible(fin.readlines())
        preprocessed_text = [el for el in preprocessor.preprocess(text) if len(el) > 6][
            :10000
        ]

    tfidf = TFIDFWrapper()
    tfidf.fit(preprocessed_text)
    word2vec = word2vec.fit(preprocessed_text, epochs=20)

    vocab = word2vec.get_vocab()
    vocab_size = len(list(vocab.keys()))

    model = Doc2Vec(
        word2vec=word2vec,
        tfidf=tfidf,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.01,
        device=torch.device("cpu"),
        embedding_dim=128,
        hidden_dim=128,
        last_n=3,
        vocab=vocab,
        vocab_size=vocab_size,
    )

    model.fit(preprocessed_text)

    model.save("model")

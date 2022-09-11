import torch
from torch import nn
import argparse

from models import Doc2VecLM, TFIDFWrapper, Word2VecWrapper
from preprocessing import Preprocessor, preprocess_bible

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--input-dir', type=str, help='Data directory path')
    parser.add_argument('--model', type=str, help='Model save path')
    args = parser.parse_args()

    preprocessor = Preprocessor(mode="train")
    word2vec = Word2VecWrapper(vector_size=128, window=5, workers=-1)

    with open(f"{args.input_dir}/train.txt", "r") as fin:
        text = preprocess_bible(fin.readlines())
        preprocessed_text = [el for el in preprocessor.preprocess(text) if len(el) > 8][:11000]

    tfidf = TFIDFWrapper()
    tfidf.fit(preprocessed_text)
    word2vec = word2vec.fit(preprocessed_text, epochs=20)

    vocab = word2vec.get_vocab()
    vocab_size = len(list(vocab.keys()))

    model = Doc2VecLM(
        word2vec=word2vec,
        tfidf=tfidf,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.0001,
        device=torch.device("cpu"),
        embedding_dim=128,
        hidden_dim=128,
        last_n=5,
        vocab=vocab,
        vocab_size=vocab_size,
    )

    model.fit(preprocessed_text)

    model.save_model(args.model)

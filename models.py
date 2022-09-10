import pickle
import random
from typing import Iterable, List, Tuple, Union

import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from tqdm import tqdm

from preprocessing import Preprocessor


class Word2VecWrapper:
    def __init__(self, vector_size=128, window=4, workers=-1):
        self.vector_size = vector_size
        self.window = window
        self.workers = workers

    def get_embeddings(self, tokens: list) -> np.ndarray:
        return self.word2vec.wv[tokens]

    def fit(self, corpus: Union[str, list], epochs: int = 20) -> "Word2VecWrapper":
        self.word2vec = Word2Vec(
            corpus,
            sg=1,
            vector_size=self.vector_size,
            window=self.window,
            min_count=1,
            workers=self.workers,
        )
        self.word2vec.train(corpus, total_examples=len(corpus), epochs=epochs)
        return self

    def get_vocab(self):
        return self.word2vec.wv.key_to_index

    def save(self, path: str) -> None:
        pass

    def __call__(self, tokens: list) -> np.ndarray:
        return self.get_embeddings(tokens)


class TFIDFWrapper:
    def __init__(self):
        self.p = Preprocessor()

    def fit(self, dataset: List) -> "TFIDFWrapper":
        self.tfidf = TfidfVectorizer(
            use_idf=True, stop_words=None, min_df=0, tokenizer=self.p.tokenize
        )
        dataset = list(map(lambda x: " ".join(x), dataset))
        self.tfidf = self.tfidf.fit(dataset)
        return self

    def get_embeddings(self, tokens: list) -> np.ndarray:
        keys = np.array(self.tfidf.get_feature_names())

        tfidf = self.tfidf.transform([" ".join(tokens)]).toarray()[0]

        indexes = np.where(tfidf > 0)[0]
        keys = keys[indexes]
        tfidf = tfidf[indexes]

        scores_dict = dict()
        for i in range(len(keys)):
            scores_dict[keys[i]] = tfidf[i]

        scores = np.array([scores_dict[token] for token in tokens])
        return scores

    def __call__(self, tokens: list) -> np.ndarray:
        return self.get_embeddings(tokens)


# TODO: separarate linear model


class Doc2Vec(nn.Module):
    def __init__(
        self,
        word2vec,
        tfidf,
        loss_function,
        optimizer,
        lr,
        device,
        vocab,
        vocab_size,
        embedding_dim,
        hidden_dim,
        last_n: int,
    ):
        super().__init__()
        self.word2vec: Word2VecWrapper = word2vec
        self.tfidf: TFIDFWrapper = tfidf
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.vocab = vocab
        self.reversed_vocab = dict()
        for key in self.vocab:
            value = self.vocab[key]
            self.reversed_vocab[value] = key
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.last_n = last_n
        self.context: np.ndarray = np.zeros(self.embedding_dim)
        self.last_n_embeddings: np.ndarray = np.zeros((self.last_n, self.embedding_dim))
        self.current_seq_len: int = 0

        self.context_linear = nn.Linear(embedding_dim, hidden_dim).float()
        self.last_n_linears = [
            nn.Linear(embedding_dim, hidden_dim).float()
        ] * self.last_n
        self.global_linear = nn.Linear(hidden_dim * (self.last_n + 1), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.vocab_size)

    def get_embeddings(self, tokens: list) -> Tuple[torch.Tensor, torch.Tensor]:
        tfidf = self.tfidf(tokens)
        embeddings = self.word2vec(tokens)

        self.last_n_embeddings = np.vstack([self.last_n_embeddings, embeddings])[-3:]
        self.new_state = np.mean(embeddings * tfidf.reshape(-1, 1), axis=0)

        new_seq_len = len(tokens)
        whole_seq_len = self.current_seq_len + new_seq_len

        last_proportion = self.current_seq_len / whole_seq_len
        new_proportion = new_seq_len / whole_seq_len

        self.context = last_proportion * self.context + new_proportion * self.new_state

        return torch.from_numpy(self.context), torch.from_numpy(self.last_n_embeddings)

    def reset(self):
        self.context: np.ndarray = np.zeros(self.embedding_dim)
        self.last_n_embeddings: np.ndarray = np.zeros((self.last_n, self.embedding_dim))
        self.current_seq_len: int = 0

    def forward(self, context, last_n_embeddings) -> torch.tensor:
        context = context.float()
        last_n_embeddings = last_n_embeddings.float()

        x_context = [self.context_linear(context)]
        x_last_n = [
            self.last_n_linears[i](last_n_embeddings[i]) for i in range(self.last_n)
        ]

        x = self.global_linear(torch.cat(x_context + x_last_n))
        probabilities = self.classifier(x)

        return probabilities

    @staticmethod
    def train_epoch(model, optimizer, loss_function, device, vocab, lines):
        model.to(device)
        model.train()

        total_loss = 0.0
        n_lines = len(lines)
        vocab_size = len(list(vocab.keys()))

        for line in tqdm(lines):
            seq_len = len(line)
            split_index = seq_len // 2
            input = line[:split_index]

            sentence_loss = 0.0
            for i in range(split_index, seq_len - 1):
                context, last_n_embeddings = model.get_embeddings(input)
                context, last_n_embeddings = context.to(device), last_n_embeddings.to(
                    device
                )
                output = model(context, last_n_embeddings)

                target = torch.zeros(vocab_size)
                target_id = torch.tensor([vocab.get(line[i])])
                target[target_id] = 1
                loss = loss_function(output, target)
                sentence_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            sentence_loss /= seq_len // 2 - 1
            total_loss += sentence_loss

        total_loss /= n_lines
        return total_loss

    @staticmethod
    def eval_epoch(model, loss_function, device, vocab, lines):
        model.to(device)
        model.eval()

        n_lines = len(lines)
        total_loss = 0.0
        vocab_size = len(list(vocab.keys()))

        for line in tqdm(lines):
            seq_len = len(line)
            split_index = seq_len // 2
            input = line[:split_index]

            sentence_loss = 0.0
            for i in range(split_index, seq_len - 1):
                context, last_n_embeddings = model.get_embeddings(input)
                context, last_n_embeddings = context.to(device), last_n_embeddings.to(
                    device
                )
                with torch.no_grad():
                    output = model(context, last_n_embeddings)

                target = torch.zeros(vocab_size)
                target_id = torch.tensor([vocab.get(line[i])])
                target[target_id] = 1
                loss = loss_function(output, target)
                sentence_loss += loss.item()

            sentence_loss /= seq_len // 2 - 1
            total_loss += sentence_loss

        total_loss /= n_lines
        return total_loss

    def fit(self, corpus):
        train_loss = self.train_epoch(
            model=self.model,
            lines=corpus[:9000],
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            device=self.device,
            vocab=self.vocab,
        )
        print("TRAIN LOSS", train_loss)

        eval_loss = self.eval_epoch(
            model=self.model,
            lines=self.preprocessed_text[9000:],
            loss_function=self.loss_function,
            device=self.device,
            vocab=self.vocab,
        )
        print("EVAL LOSS", eval_loss)

    def generate(self, tokens: list = None, seq_len: int = None) -> str:
        context, last_n_embeddings = self.get_embeddings(tokens)

        if tokens is None:
            tokens = ["pad", "pad", "bos"]
            tokens += [self.reversed_vocab[random.randint(0, self.vocab_size - 1)]]

        if seq_len:
            for i in range(seq_len):
                id = torch.argmax(self.forward(context, last_n_embeddings))
                word = self.reversed_vocab[id]
                tokens += [word]
                context, last_n_embeddings = self.get_embeddings([word])
            return " ".join(tokens)
        else:
            if seq_len:
                for i in range(seq_len):
                    id = torch.argmax(self.forward(context, last_n_embeddings))
                    word = self.reversed_vocab[id]
                    if word == "eos":
                        return " ".join(tokens)
                    tokens += [word]
                    context, last_n_embeddings = self.get_embeddings([word])

    def save_model(self, name: str) -> None:
        with open(f"{name}.pkl", "wb") as fout:
            pickle.dump(self, file=fout)

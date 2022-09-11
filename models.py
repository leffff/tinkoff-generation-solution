import pickle
import random
from typing import Any, List, Tuple, Union

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
            use_idf=True, stop_words=None, min_df=0, tokenizer=self.p.tokenize, token_pattern=None
        )
        dataset = list(map(lambda x: " ".join(x), dataset))
        self.tfidf = self.tfidf.fit(dataset)
        return self

    def get_embeddings(self, tokens: list) -> np.ndarray:
        keys = np.array(self.tfidf.get_feature_names_out())

        tfidf = self.tfidf.transform([" ".join(tokens)]).toarray()[0]

        indexes = np.where(tfidf > 0)[0]
        keys = keys[indexes]
        tfidf = tfidf[indexes]

        scores_dict = dict()
        for i in range(len(keys)):
            if keys[i] == "pad":
                scores_dict[keys[i]] = 0
            else:
                scores_dict[keys[i]] = tfidf[i]

        scores = np.array([scores_dict[token] for token in tokens])
        return scores

    def __call__(self, tokens: list) -> np.ndarray:
        return self.get_embeddings(tokens)


class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, last_n, vocab_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.last_n = last_n
        self.vocab_size = vocab_size

        self.context_linear = nn.Linear(embedding_dim, hidden_dim)
        self.last_n_linears = [nn.Linear(embedding_dim, hidden_dim)] * self.last_n
        self.global_linear = nn.Linear(hidden_dim * (self.last_n + 1), hidden_dim)
        self.classifier = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, context, last_n_embeddings) -> torch.tensor:
        context = context.float()
        last_n_embeddings = last_n_embeddings.float()

        x_context = [self.context_linear(context)]
        x_last_n = [self.last_n_linears[i](last_n_embeddings[i]) for i in range(self.last_n)]

        x = self.global_linear(torch.cat(x_context + x_last_n))
        probabilities = self.classifier(x)

        return probabilities


class Doc2VecLM:
    def __init__(
            self,
            word2vec: Word2VecWrapper = None,
            tfidf: TFIDFWrapper = None,
            loss_function: nn.Module = None,
            optimizer: Any = None,
            lr: float = None,
            device: torch.device = None,
            vocab: dict = None,
            vocab_size: int = None,
            embedding_dim: int = None,
            hidden_dim: int = None,
            last_n: int = None,
    ):
        super().__init__()
        self.word2vec: Word2VecWrapper = word2vec
        self.tfidf: TFIDFWrapper = tfidf
        self.classifier = Classifier(embedding_dim, hidden_dim, last_n, vocab_size)

        self.loss_function = loss_function
        self.lr = lr
        self.optimizer = optimizer(self.classifier.parameters(), lr=self.lr)
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

    def get_embeddings(self, tokens: list) -> Tuple[torch.Tensor, torch.Tensor]:
        tfidf = self.tfidf(tokens)
        embeddings = self.word2vec(tokens)

        self.last_n_embeddings = np.vstack([self.last_n_embeddings, embeddings])[-self.last_n:]
        self.new_state = np.mean(embeddings * tfidf.reshape(-1, 1), axis=0)
        # self.new_state = np.mean(embeddings, axis=0)

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

    def train_epoch(self, model, optimizer, loss_function, device, vocab, lines):
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
                context, last_n_embeddings = self.get_embeddings(input)
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
            self.reset()

        total_loss /= n_lines
        return total_loss

    def eval_epoch(self, model, loss_function, device, vocab, lines):
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
                context, last_n_embeddings = self.get_embeddings(input)
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
            self.reset()

        total_loss /= n_lines
        return total_loss

    def fit(self, corpus, eval_percent: float = 0.2) -> "Doc2VecLM":
        total_examples = len(corpus)
        split_int = int(total_examples * (1 - eval_percent))

        train_loss = self.train_epoch(
            model=self.classifier,
            lines=corpus[:split_int],
            optimizer=self.optimizer,
            loss_function=self.loss_function,
            device=self.device,
            vocab=self.vocab,
        )
        print("TRAIN LOSS", train_loss)

        eval_loss = self.eval_epoch(
            model=self.classifier,
            lines=corpus[split_int:],
            loss_function=self.loss_function,
            device=self.device,
            vocab=self.vocab,
        )
        print("EVAL LOSS", eval_loss)

        return self

    def generate(self, tokens: list = None, seq_len: int = None) -> str:
        if tokens is None:
            tokens = ["pad"] * self.last_n
            tokens = tokens + ["bos"]
            tokens += [self.reversed_vocab[random.randint(0, self.vocab_size - 1)]]

        context, last_n_embeddings = self.get_embeddings(tokens)

        if seq_len:
            for i in range(seq_len):
                probas = self.classifier(context, last_n_embeddings).softmax(0)
                probas, indices = torch.sort(probas)
                random_samples = 20
                id = indices[-random_samples:][random.randint(0, random_samples - 1)].item()
                word = self.reversed_vocab[id]
                while word in tokens:
                    id = indices[-random_samples:][random.randint(0, random_samples - 1)].item()
                    word = self.reversed_vocab[id]

                tokens += [word]
                context, last_n_embeddings = self.get_embeddings([word])
            return " ".join(tokens)
        else:
            for i in range(seq_len):
                probas = self.classifier(context, last_n_embeddings).softmax(0)
                probas, indices = torch.sort(probas)
                random_samples = 20
                id = indices[-random_samples:][random.randint(0, random_samples - 1)].item()
                word = self.reversed_vocab[id]
                while word in tokens:
                    id = indices[-random_samples:][random.randint(0, random_samples - 1)].item()
                    word = self.reversed_vocab[id]
                if word == "eos":
                    return " ".join(tokens)
                tokens += [word]
                context, last_n_embeddings = self.get_embeddings([word])

    def save_model(self, path_to_model: str) -> None:
        if not path_to_model.endswith("pkl"):
            raise ValueError("Model extension must be .pkl")

        with open(f"{path_to_model}", "wb") as fout:
            pickle.dump(self, file=fout)

    @staticmethod
    def load_model(path_to_model: str) -> "Doc2VecLM":
        if not path_to_model.endswith("pkl"):
            raise ValueError("Model extension must be .pkl")

        with open(f"{path_to_model}", "rb") as fin:
            model = pickle.load(file=fin)

        return model
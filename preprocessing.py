from gensim.utils import tokenize


class Preprocessor:
    def __init__(self, mode: str = "test", padding_size=3):
        self.mode = mode.lower()
        self.padding_size = padding_size

    @staticmethod
    def delete_punct(sentence: str) -> str:
        sentence = "".join([c for c in sentence if c == " " or c.isalpha()])

        return sentence

    @staticmethod
    def tokenize(line: str) -> list:
        tokenized_line = list(tokenize(line, lowercase=True, deacc=True))

        return tokenized_line

    def add_special_tokens(self, line: list) -> list:
        for i in range(self.padding_size):
            line.insert(0, "pad")

        line.insert(2, "bos")
        if self.mode == "train":
            line.insert(len(line), "eos")
        return line

    def preprocess(self, sentences: list) -> list:
        methods = [self.delete_punct, self.tokenize, self.add_special_tokens]

        for method in methods:
            sentences = list(map(method, sentences))

        return sentences


def preprocess_bible(lines: list) -> list:
    new_lines = []
    for line in lines:
        if line != "":
            new_lines.append(" ".join(line.split()[1:]))

    return new_lines

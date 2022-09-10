from gensim.utils import tokenize


class Preprocessor:
    @staticmethod
    def delete_punct(sentence: str) -> str:
        sentence = "".join([c for c in sentence if c == " " or c.isalpha()])

        return sentence

    @staticmethod
    def tokenize(line: str) -> list:
        tokenized_line = list(tokenize(line, lowercase=True, deacc=True))

        return tokenized_line

    @staticmethod
    def add_special_tokens(line: list) -> list:
        line.insert(0, "pad")
        line.insert(1, "pad")
        line.insert(2, "bos")
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

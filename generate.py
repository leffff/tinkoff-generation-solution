from models import Doc2VecLM
from preprocessing import Preprocessor

if __name__ == "__main__":
    preprocessor = Preprocessor(mode="test")

    with open("data/test.txt", "r") as fin:
        text = fin.readlines()
        preprocessed_text = preprocessor.preprocess(text)

    # print(preprocessed_text[0])
    model = Doc2VecLM.load_model("model.pkl")

    print(model.generate(seq_len=10))

import argparse

from models import Doc2VecLM
from preprocessing import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generation')
    parser.add_argument('--model', type=str, help='Model load path')
    parser.add_argument('--prefix', type=str, default=None, help='Input to model')
    parser.add_argument('--length', type=int, default=5, help='Generation length')
    args = parser.parse_args()

    preprocessor = Preprocessor(mode="test")

    text = None
    if args.prefix:
        text = preprocessor.preprocess([args.prefix])[0]

    model = Doc2VecLM.load_model(args.model)

    print(model.generate(text, seq_len=args.length))

import sys
sys.path.append(r"/home/jungo/code/AT3DCV2022/1++_FINAL")  # set path to the directory

from evaluation import Evaluation


def main():
    eval = Evaluation()
    eval.load_mono_model()
    eval.test()


if __name__ == "__main__":
    main()

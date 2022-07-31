from eval_pointcloud import Trainer

def main():
    eval = Trainer()
    eval.load_mono_model()
    eval.test()


if __name__ == "__main__":
    main()
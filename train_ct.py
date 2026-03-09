from lung_cancer_cls.train import build_parser, train_main


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_main(args)

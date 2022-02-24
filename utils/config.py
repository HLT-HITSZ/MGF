import argparse

def get_config():
    config = argparse.ArgumentParser()
    config.add_argument("--data-path", default="./data/PE_data_df.csv", type=str)
    config.add_argument("--split-test-file-path", default="./data/test_paragraph_index.json", type=str)
    config.add_argument("--split-dev-file-path", default="./data/dev_essay/dev_essay_index.json", type=str)
    config.add_argument("--vocab-path", default="./data/bow_vocab.json", type=str)
    config.add_argument("--save-path", default="./saved_models", type=str)
    config.add_argument("--bert-path", default="./data/bert-base-uncased", type=str)

    # training
    config.add_argument("--device", default='1', type=str)
    config.add_argument("--seed", default=42, type=int)
    config.add_argument("--batch-size", default=2, type=int)
    config.add_argument("--epochs", default=20, type=int)
    config.add_argument("--showtime", default=1816, type=int)
    config.add_argument("--base-encoder-lr", default=1e-5, type=float)
    config.add_argument("--finetune-lr", default=1e-3, type=float)
    config.add_argument("--warm-up", default=5e-2, type=float)
    config.add_argument("--weight-decay", default=1e-5, type=float)
    config.add_argument("--early-num", default=1, type=int)
    config.add_argument("--num-tags", default=5, type=int)
    config.add_argument("--threshold", default=0.0, type=float)
    config.add_argument("--am-weight", default=0.5, type=float)

    # trans model param
    config.add_argument("--hidden-size", default=256, type=int)
    config.add_argument("--layers", default=1, type=int)
    config.add_argument("--is-bi", default=True, type=bool)
    config.add_argument("--bert-output-size", default=768, type=int)
    config.add_argument("--mlp-size", default=512, type=int)
    config.add_argument("--scale-factor", default=2, type=int)
    config.add_argument("--dropout", default=0.5, type=float)
    config.add_argument("--max-grad-norm", default=1.0, type=float)
    config = config.parse_args()

    return config
from  factories.sns_raw_prep_sep_dnn_factory import SNSRawPrepSepDNNFactory
# --------------------------
#  MAIN with ARGPARSE
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE-BiLSTM Pipeline for SNS Data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for SGD optimizer")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension for VAE")
    args = parser.parse_args()

    factory = SNSRawPrepSepDNNFactory()
    factory.run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim
    )
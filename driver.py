from  factories.sns_raw_prep_sep_dnn_factory import SNSRawPrepSepDNNFactory
import argparse

"""
Example: SNSRawPrepSepDNNFactory with subcommands (train/predict), 
refactored to reduce repeated data prep code.

Usage:
  # Train:
  python driver.py train \
    --epochs 100 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 \
    --model_path saved_models/vae_bilstm_model.h5

  # Predict:
  python driver.py predict \
    --model_path saved_models/vae_bilstm_model.h5 --threshold_percentile 90
"""


# --------------------------
# MAIN with ARGPARSE SUBCOMMANDS
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="VAE-BiLSTM Pipeline with train/predict, minimal code repetition")
    subparsers = parser.add_subparsers(dest="command", help="train or predict")

    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train the VAE-BiLSTM model")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    train_parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for SGD")
    train_parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension for VAE")
    train_parser.add_argument("--model_path", type=str, default="vae_bilstm_model.h5", help="Where to save the model weights")
    train_parser.add_argument("--tensorboard_logdir", type=str, default="logs/fit", help="Base directory for TensorBoard logs")

    # Subcommand: predict
    predict_parser = subparsers.add_parser("predict", help="Use a trained model to predict anomalies")
    predict_parser.add_argument("--model_path", type=str, default="vae_bilstm_model.h5", help="Path to saved model weights")
    predict_parser.add_argument("--threshold_percentile", type=float, default=95.0, help="Percentile for anomaly threshold")

    args = parser.parse_args()

    factory = SNSRawPrepSepDNNFactory()

    if args.command == "train":
        factory.train_pipeline(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            latent_dim=args.latent_dim,
            model_path=args.model_path,
            tensorboard_logdir=args.tensorboard_logdir 
        )
    elif args.command == "predict":
        factory.predict_pipeline(
            model_path=args.model_path,
            threshold_percentile=args.threshold_percentile
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
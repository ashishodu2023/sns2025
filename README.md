

### PipeLine 

  # Train:
  python driver.py train \
    --epochs 5 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 \
    --model_path vae_bilstm_model.weights.h5 \
    --tensorboard_logdir logs/fit

  # Predict:
  python driver.py predict \
    --model_path vae_bilstm_model.weights.h5 --threshold_percentile 90

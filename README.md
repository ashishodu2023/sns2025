

### PipeLine 

  # Train:
  python driver.py train \
    --epochs 100 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 \
    --model_path saved_models/vae_bilstm_model.h5

  # Predict:
  python driver.py predict \
    --model_path saved_models/vae_bilstm_model.h5 --threshold_percentile 90

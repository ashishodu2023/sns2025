

### PipeLine 

  # Train:
nohup python driver.py train \
    --epochs 5 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 \
    --model_path saved_models/vae_bilstm_model.weights.h5 \
    --tensorboard_logdir logs/fit >& train.log &



  # Predict:
  python driver.py predict \
    --model_path /u/home/averm/sns2025/saved_models/vae_bilstm_model.weights.h5 --threshold_percentile 99



nohup /u/home/averm/sns2025/sns_raw_prep_sep_cnn_lstm.py >& cnn_lstm_train.log &
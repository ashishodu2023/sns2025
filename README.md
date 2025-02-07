

### Training 
* python driver.py --model Autoencoder --path models/autoencoder.h5 --mode train --epochs 20 --batch_size 64

### Inference 
* python driver.py --model Autoencoder --path models/autoencoder.h5 --mode predict 

### Benchmarking 
* python driver.py --model Autoencoder --path models/autoencoder.h5 --mode benchmark --device CPU
* python driver.py --model Autoencoder --path models/autoencoder.h5 --mode benchmark --device GPU


### Realtime 
* python driver.py --model Autoencoder --path models/autoencoder.h5 --mode anomaly --steps 100

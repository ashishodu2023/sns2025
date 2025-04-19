<div align="center">
<img src="https://www.odu.edu/sites/default/files/logos/univ/png-72dpi/odu-sig-noidea-fullcolor.png" style="width:225px;">
</div>

<div align="center"> <font color=#003057>
        
# Data Science Capstone Project Spring 2025 

</font>

<div> 
<font size=4 color=#828A8F><b>May 2025</b></font><br>
<font size=4><i>AJ Broderick, Arun Thakur, Ashish Verma</i></font>
</div>

</div>

## <font color=#003057> ***Findings & Lessons Learned*** </font>

### <u>Binary Transformation & Data Size</u> 
&emsp; Early on, and throughout, the project the team discovered the challenge of working with data that is stored in a binary format. This deviates from *"traditional"* data sources that might be stored in tables or flat files that could be read into Python and stored in DataFrames. Because of this we had to think of ways to preprocess as much of the data as possible to reduce the time it would take to extract, transform and store the beam tracing data. A technique that we tested was to utilize the beam parameters to assist in filtering out data that might clog up the models. This was done by removing files that were associated with beams that were not in production based on the value of the `'ICS_Tim:Gate_BeamOn:RR'` column being below 59.9Hz. Also by grouping files based on the same beam configurations, we could filter files and resulting tracings on groups that did not have many samples based on fine tuning the beam. 

### <u>Class Imbalance</u>
&emsp; Even after the tests and implementation of the filtering techniques mentioned above, there was another factor of the data that we had to address. This was from nature of how the data was collected from the SNS, and there exists more data for when things are operating normally and when faults occur. 


### <u>Model Overview & Findings</u>
&emsp; Throughout the semester the team looked to implement more than one model to attempt to detect the anomalies. We tried to develop techniques that had not been documented as machine learning models that have been applied to the SNS data, to give a fresh perspective of insights. From this two models were developed and tested, an unsupervised model and a supervised model. The next section gives a high level description of the models and explores insights from the outputs.

#### ***VAE-BiLSTM***
&emsp; The first model we explored was unsupervised, and utilized a blended model to process the data for anomaly detection, which was a Variable Autoencoder with Bilaterial Long-Short Term Memory. During initial research for machine learning models that excel in anomaly detection, the team came upon a paper for a team that used this model within the medical sector to detect anomalies in heart rates via data collected from wearable devices <sup>[1](https://doi.org/10.3390/bioengineering10060683)</sup>. The data that is collected from the SNS had patterns that were repeatable with each run, similar to how the heart rate data would come through, and determined that the model could generate insights on these patterns.\
&emsp; The two segments of the VAE-BiLSTM model both aided in anomaly detection. Variable Autoencoders learn from a latent representation of data and attempts to reconstruct it, and a BiLSTM is a recurrent neural network that processes data in both forward and backward directions, capturing context from both past and future states. By combining these with the time series data of the SNS, the model would be able to look forward and backwards through the data as it attempts to reconstruct the latent data to determine if it was normal.\

#### ***CNN LSTM***
&emsp; The second model we explored was supervised and again utilized a blended model, which was a Convolutional Neural Network with Long-Short Term Memory <sup>[2](https://doi.org/10.1145/3465481.3469190)</sup>. With this model, we were able to create labels for the data based on whether it was extracted from the normal file(0) or the anomalous file(1). By setting this value as the y value of the model, once the data was processed it would classify the results into this binary format. A CNN LSTM model is another strong model choice for anomaly detection from the temporal aspect of the SNS data. CNN layers extract features from chunks of input data whilst the LSTM layers model the temporal relationships between these extracted features and are good for detecting subtle changes that indicate an anomaly.\

### <u>Individual Insights</u>

**AJ**:\
&emsp; For myself, this is the first data science project that I had a chance to work on and it was good exposure to how projects operate in application. From a data analysis standpoint, it was interesting to see how data science and machine learning could be applied to data that had primarily numeric input/outputs. It had me pushing past some of the basic transformations of the data to try to extract something that was meaningful and actionable. From the machine learning side of things, it was good to go through the MLOps of designing the models that we used, developing the models and then testing them. Even as we approach the end of our time with the JLab, we thoughts on how could be iteratively change the models to refine the output for greater insights.\
&emsp; Moreover I gained a lot of knowledge and experience from the coding side of the project. I come from an analytics background that is SQL-based and throughout my time at ODU have had to utilize Python. However, with the sample code that was provided by the JLab and other members of my team, I was able to see how Python is used at a higher degree. From this, I hope to take some of these teachniques and apply them to projects that I may work on in the future. 

**Arun**:\
The Anomaly Prediction at the Spallation Neutron Source Accelerator project explores machine learning techniques to detect errant beam pulses, ensuring efficient accelerator operations. Here are some key lessons learned:

1. Data Preprocessing
Feature Engineering: Selecting relevant features from accelerator sensor data significantly improves model performance.

Normalization: Standardizing input data enhances stability, especially for deep learning models like VAE and BiLSTM.

Handling Missing Data: Imputation techniques help maintain data integrity, preventing bias in anomaly detection.

2. Model Selection
VAE (Variational Autoencoder): Useful for learning latent representations and detecting anomalies based on reconstruction errors.

BiLSTM (Bidirectional Long Short-Term Memory): Effective for capturing temporal dependencies in accelerator data.

Hybrid Approaches: Combining VAE and BiLSTM improves robustness in anomaly detection.

3. Performance Accuracy
Evaluation Metrics: Precision, recall, and F1-score are crucial for assessing anomaly detection effectiveness.

Threshold Optimization: Fine-tuning anomaly detection thresholds minimizes false positives and false negatives.

4. Class Imbalance
Synthetic Data Generation: Techniques like SMOTE help balance rare anomaly cases.

Weighted Loss Functions: Adjusting loss functions ensures the model prioritizes minority class detection.

5. Hyperparameter Tuning
Grid Search & Bayesian Optimization: Used to optimize learning rates, batch sizes, and network architectures.

Regularization Techniques: Dropout and batch normalization prevent overfitting.

**Ashish**:\

* Strict Data Separation

  - Always split into train/validation/test before any oversampling or transformation to avoid data leakage.

  - Apply SMOTE (or any augmentation) only on the training set.

* Balancing Precision and Recall

  - Default 0.5 thresholds can be too conservative for rare‐event detection.

  - Lowering your decision threshold can substantially boost recall at the expense of a controlled rise in FPR.

* Class Imbalance Strategies

  - Use class weights in your loss function to penalize missed anomalies more heavily.

  - Experiment with both oversampling (SMOTE) and undersampling, tuning the sampling ratio and neighbor parameters.

* Regularization and Overfitting Control

  - Add L2 weight penalties and dropout (including recurrent dropout in LSTMs) to discourage memorization.

  - Monitor training vs. validation losses and use early stopping or learning‑rate schedules.

* Hyperparameter Tuning & Model Complexity

  - Perform systematic searches (grid/random/Bayesian) over parameters like tree depth, learning rate, dropout rate, L2 strength, and network size.

  - Ensure your model has enough capacity to learn anomalies—but not so much that it overfits the majority class.

* Feature Engineering

  - Go beyond basic statistics on your “traces”: extract frequency‑domain features (FFT, wavelets), autocorrelations, rolling‑window moments, skewness, kurtosis, etc.

  - Leverage domain knowledge to craft features that highlight known anomaly signatures.

* Alternative Anomaly Detection Techniques

  - Evaluate specialized unsupervised methods (Isolation Forest, One‑Class SVM, autoencoders) that focus on modeling “normal” and flag deviations.

  - Consider ensemble or stacked models to combine different algorithmic strengths.

* Probability Calibration & Evaluation

  - Use Platt scaling or isotonic regression to align predicted probabilities with true event likelihoods.

  - Rigorously assess performance via k‑fold cross‑validation, ROC/PR curves, and domain‑specific cost metrics to choose the best operating poin

## <font color=#003057>***Recommendations***</font>

### <u>Data Access</u> 
&emsp; One of the first hurdles that the team came across was accessing the data. The Spring Semester started on January 11th and after initial meetings with the Jefferson Lab and waiting on clearance checks, we did not get access to the data until late-January/early-February. This resulted in a loss of at least three weeks to work on the project.

<font color=#4348DD>

  * The team's recommendation for future capstone projects would be to get the submission of documents and required IT trainings in Week 1. If an SOP specifically to students working with the Jefferson Lab could be developed containing the different steps and requirements needed, it could be distributed as soon as the teams are developed. This would speep up timing of getting students into the data, and give more time for data analysis and model development

</font>

### <u>iFarm & slurm</u>
&emsp; Similiar to accessing the data, one challenge that the team faced was running large scale models in the JLab environment once the models were developed in the Jupyter Notebooks. There was some trial and error that occurred when attempting to get the environment up and running in which to execute the code. Kishan did a great job in finding a solution that worked and in providing some documentation once issues were resolved 

<font color=#4348DD>
        
  * Expanding on the previous recommendation of an SOP for ODU students, there should be a JLab version that walks through the steps that would be required to create a shared folder and the required code/sub-folders for the teams to execute the code
  * Another thing that would be benefical for future ODU students would be guidelines and explinations on bits of code that we're able to change for submitting batches to slurm. We were hesitant to change too much to avoid causing downstream impacts on JLab processing cores by accidently overindexing on resources
    
</font>

## <font color=#003057> ***Model Notes*** </font>

### <font color=#98C5EA>**Requirements**</font>
```bash
Python==3.9.21
numpy==2.0.2
matplotlib==3.9.4
seaborn==0.13.2
pandas==2.2.3
imblearn==0.0
tensorflow==2.18.0
tensorflow-estimator==2.13.0
tensorflow-io-gcs-filesystem==0.37.1
python-dateutil==2.9.0.post0
scikit-learn==1.6.1
scipy==1.13.1
```

### <font color=#98C5EA>**VAE-BiLSTM**</font>

#### *Structure*

```    
sns_2025
├── config
│   ├── bpm_config.py
│   └── dcm_config.py
├── data
│   ├── data_preprocessor.py
│   ├── beam_data_loader.py
│   └── merge_datasets.py
├── factories
│   ├── sns_raw_prep_sep_dnn_factory.py
│   └── sns_raw_prep_sep_dnn_factory_updated.py
├── models
│   └── vae_bilstm.py
├── utils
│   └── logger.py
├── visualizations
│   └── plots.py
├── driver.py
└── requirements.txt
```

#### *Installation & Execution*
``` bash
#Install
git clone sns_2025
cd sns_2025
pip install -r requirements.txt

# Train:
python driver.py train --epochs 5 --batch_size 8 --learning_rate 1e-4 --latent_dim 32 --model_path vae_bilstm_model.weights.h5 --tensorboard_logdir logs/fit

# Predict:
python driver.py predict --model_path vae_bilstm_model.weights.h5 --threshold_percentile 90
```

### <font color=#98C5EA>**[CNN LSTM](https://github.com/ashishodu2023/sns_cnn_lstm)**</font>

#### *Structure*

```         
sns_cnn_lstm
├── analysis
│   └── evaluation.py
├── data_preparation
│   ├── data_preprocessor.py
│   └── data_scaling.py
├── model
│   └── anomaly_model.py
├── parser
│   ├── bpm_parser.py
│   └── dcm_parser.py
├── train_flow.py
├── test_flow.py
├── main.py
└── requirements.txt
```

#### Installation & Execution

``` bash
# Install
git clone sns_cnn_lstm
cd sns_cnn_lstm
pip install -r requirements.txt

# Train:
python main.py --train

# Test:
python main.py --test
```

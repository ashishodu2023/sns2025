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


### <u>Model Findings</u>
&emsp; Place holder to talk about development of the unsupervised and supervised models  

* **VAE-BiLSTM**
* **CNN LSTM**

### <u>Individual Insights</u>

**AJ**:\
&emsp; For myself, this is the first data science project that I had a chance to work on and it was good exposure to how projects operate in application. From a data analysis standpoint, it was interesting to see how data science and machine learning could be applied to data that had primarily numeric input/outputs. It had me pushing past some of the basic transformations of the data to try to extract something that was meaningful and actionable. From the machine learning side of things, it was good to go through the MLOps of designing the models that we used, developing the models and then testing them. Even as we approach the end of our time with the JLab, we thoughts on how could be iteratively change the models to refine the output for greater insights.\
&emsp; Moreover I gained a lot of knowledge and experience from the coding side of the project. I come from an analytics background that is SQL-based and throughout my time at ODU have had to utilize Python. However, with the sample code that was provided by the JLab and other members of my team, I was able to see how Python is used at a higher degree. From this, I hope to take some of these teachniques and apply them to projects that I may work on in the future. 

**Arun**:\
&emsp; Lorem Ipsum

**Ashish**:\
&emsp; Lorem Ipsum

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

### <font color=#98C5EA>**CNN LSTM**</font>

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

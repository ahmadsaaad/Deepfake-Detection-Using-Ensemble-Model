This repository is using the code from [A Comprehensive Benchmark of Deepfake Detection](https://github.com/SCLBD/DeepfakeBench/tree/main) repository. 

## Dataset

The dataset used for this project is [FaceForensics++](https://github.com/ondyari/FaceForensics)
To prepare the dataset, Please follow the steps below:

 - Download the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset (Make sure you are downloading the c23 version)
 - Extraxt the Dataset under the < the project folder>/datasets
 - As we are going to use Deepfakes sub-dataset for training and NeuralTextures sub-dataset for testing, you may remove other folders under the manipulated_sequences folder to save space **ONLY AFTER RUNNING COMPLETING THE PREPROCESSING STEPS**.
 - Once you extract the dataset folder you should have < the project folder>/datasetsFaceForensics++ which contains 2 folders original_sequences and manipulated_sequences
 - Copy json files test, train and val from < the project folder>/datasets to < the project folder>/datasetsFaceForensics++ 

## Preprocessing
For data preprocessing steps and training process, pleaes follow the steps in the [A Comprehensive Benchmark of Deepfake Detection](https://github.com/SCLBD/DeepfakeBench/tree/main) repository.
In config.yaml, make sure that the default dataset is set to 'FaceForensics++' for both preprocess & rearrange. 

## Training

Before running the training, you should first download the pretrained weights for the corresponding backbones. You can download them from this [link](https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip). After downloading, you need to put all the weights files into the folder ./training/pretrained/. 
(Please go through the steps in pleaes follow the steps in the [A Comprehensive Benchmark of Deepfake Detection](https://github.com/SCLBD/DeepfakeBench/tree/main)  training section to learn more)

As we are using the following detectors in our ensemble:

 1. Capsule
 2. CORE
 3. DSP-FWA

You may need to run the training for these 3 models only. When you run the training for this models, please make sure that you set the train_dataset in the corespending yaml file to ['FF-DF']

> Training process can take several days if run on CPU. So, please make sure to start the training process on a machine that has a GPU to make the traininng process faster. 

After the completion of the training, pleae follow the steps below: 
1. Create folder under your project folder called models. 
2. Under this folder create 3 folders with names (capsule, core, fwa) **(All Small letters)**
3. copy ckpt_best.pth from each output training folder of each detector to the corresponding folder above. 

## Testing

1. Open  <Project folder/training>/training/test_and_store_results_all_models.py
2. Modify model_pat with the path to the parent folder that you created after the training completed. 
3. Modify the detector paths with your local paths. 
4. Make sure that the test_dataset vairable set to ['FF-NT']
5. update the cpu_device & mps_device variables as follow:
	a.  If you are running the test on a windows/linux pc that doesn't have cuda installed, please replace mps with cpu
	b. If you are running on a mac machine that supports mps, please leave MPS 
6. Now go to your terminal and type `cd  <Project folder/training>/training/`
7. then `python test_and_store_results_all_models.py`
8. After the test got completed, you should have 3 pkl files under <Project folder/training>/training/models-obj

## Run The Ensemble Model 

1. <Project folder/training>/training/build_test_ensemble.py
2. Modify model_pat with the path to the parent folder that you created after the training completed. 
3. Modify the detector paths with your local paths. 
4. Make sure that the test_dataset vairable set to ['FF-NT']
5. update the cpu_device & mps_device variables as follow:
	a.  If you are running the test on a windows/linux pc that doesn't have cuda installed, please replace mps with cpu
	b. If you are running on a mac machine that supports mps, please leave MPS 
6. Now go to your terminal and type `cd  <Project folder/training>/training/`
7. then `python build_test_ensemble.py`
8. You will get the results printed on the screen.

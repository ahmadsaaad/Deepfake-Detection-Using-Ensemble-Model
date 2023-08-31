"""
eval pretained model.
"""

import numpy as np
import random
import yaml
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

from dataset.test_dataset import testDataset

from detectors import DETECTOR
from sklearn import metrics
import os
import pickle




mps_device = torch.device("cuda" if torch.cuda.is_available() else "mps")
cpu_device=torch.device("cuda" if torch.cuda.is_available() else "mps")
model_pat='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training-resilts/models'


ensemble_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/ensemble.yaml'

capsule_net_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/capsule_net.yaml'
capsule_net_model_path=model_pat+'/capsule_net/ckpt_best.pth'


core_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/core.yaml'
core_model_path=model_pat+'/core/ckpt_best.pth'


ffd_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/ffd.yaml'
ffd_model_path=model_pat+'/ffd/ckpt_best.pth'


fwa_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/fwa.yaml'
fwa_model_path=model_pat+'/fwa/ckpt_best.pth'




test_dataset=['FF-NT']




def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = [test_name]  # specify the current test dataset
        test_set = testDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    for i, data_dict in tqdm(enumerate(data_loader)):
        # get data
        data, label, label_spe, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['label_spe'], data_dict['mask'], data_dict['landmark']
    
        # move data to GPU
        data_dict['image'], data_dict['label'], data_dict['label_spe'] = data.to(mps_device), label.to(mps_device), label_spe.to(mps_device)
        if mask is not None:
            data_dict['mask'] = mask.to(mps_device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(mps_device)
        inference(model, data_dict)

def test_epoch(models, test_data_loaders):
    # set model to eval mode
    
    for md in models:
        md.eval()


    keys = test_data_loaders.keys()
    ensemble_probs_accumulator=[]
    
    for key in keys:
        
        for model in models:
            model_filename = os.path.join("models-obj", f"{key}_{model.__class__.__name__}.pkl")
            
            try:
                
                if os.path.exists(model_filename):
                    print(f"Loading model from {model_filename}")
                    loaded_model = load_model_from_pickle(model_filename)
                    model = loaded_model
                else:
                    test_one_dataset(model, test_data_loaders[key])
                    # Save the generated model to a pickle file
                    save_model_to_pickle(model, model_filename)
            except Exception as e:
                # Handle the exception
                print(e)
                pass
            else:
                y_prob_model = np.concatenate(model.prob)
                ensemble_probs_accumulator.append(y_prob_model)
                y_true = np.concatenate(model.label)
                
        ensemble_probs = np.mean(ensemble_probs_accumulator, axis=0)
        ensemble_fpr, ensemble_tpr, ensemble_thresholds = metrics.roc_curve(y_true, ensemble_probs, pos_label=1)
        ensemble_auc = metrics.auc(ensemble_fpr, ensemble_tpr)
        print('Ensemble AUC:', ensemble_auc)


    
def save_model_to_pickle(model, filename):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)

def load_model_from_pickle(filename):
    with open(filename, 'rb') as pickle_file:
        loaded_model = pickle.load(pickle_file)
    return loaded_model
@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # Ensemble
    with open(ensemble_detector_path, 'r') as f:
        ensemble_config = yaml.safe_load(f)
    ensemble_config['test_dataset'] = test_dataset
    # set cudnn benchmark if needed
    if ensemble_config['cudnn']:
        cudnn.benchmark = True
    # init seed
    init_seed(ensemble_config)
    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(ensemble_config)
    
    
        
    with open(capsule_net_detector_path, 'r') as f:
        capsule_net_config = yaml.safe_load(f)
    capsule_net_config['weights_path'] = capsule_net_model_path
    # # prepare the model (capsule_net_detector)
    capsule_net_model_class = DETECTOR[capsule_net_config['model_name']]
    capsule_net_model = capsule_net_model_class(capsule_net_config).to(cpu_device)
    capsule_net_ckpt = torch.load(capsule_net_model_path, map_location=cpu_device)
    capsule_net_model.load_state_dict(capsule_net_ckpt, strict=True)
    print('===> Load capsule_net checkpoint done!')
        
    with open(core_detector_path, 'r') as f:
          core_config = yaml.safe_load(f)
    core_config['weights_path'] = core_model_path
    # # prepare the model (core_detector)
    core_model_class = DETECTOR[core_config['model_name']]
    core_model = core_model_class(core_config).to(cpu_device)
    core_ckpt = torch.load(core_model_path, map_location=cpu_device)
    core_model.load_state_dict(core_ckpt, strict=True)
    print('===> Load core checkpoint done!')
  
    with open(fwa_detector_path, 'r') as f:
        fwa_config = yaml.safe_load(f)
    fwa_config['weights_path'] = fwa_model_path
    # # prepare the model (fwa_detector)
    fwa_model_class = DETECTOR[fwa_config['model_name']]
    fwa_model = fwa_model_class(fwa_config).to(cpu_device)
    fwa_ckpt = torch.load(fwa_model_path, map_location=cpu_device)
    fwa_model.load_state_dict(fwa_ckpt, strict=True)
    print('===> Load fwa checkpoint done!')

    models=[capsule_net_model,core_model,fwa_model]
    # # start training
    test_epoch(models, test_data_loaders)
  

if __name__ == '__main__':
    main()

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


efficientnetb4_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/efficientnetb4.yaml'
efficientnetb4_model_path=model_pat+'/efficientnetb4/ckpt_best.pth'



ffd_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/ffd.yaml'
ffd_model_path=model_pat+'/ffd/ckpt_best.pth'


fwa_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/fwa.yaml'
fwa_model_path=model_pat+'/fwa/ckpt_best.pth'


meso4_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/meso4.yaml'
meso4_model_path=model_pat+'/meso4/ckpt_best.pth'


meso4Inception_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/meso4Inception.yaml'
meso4Inception_model_path=model_pat+'/meso4Inception/ckpt_best.pth'


recce_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/recce.yaml'
recce_model_path=model_pat+'/recce/ckpt_best.pth'

resnet34_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/resnet34.yaml'
resnet34_model_path=model_pat+'/resnet34/ckpt_best.pth'

srm_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/srm.yaml'
srm_model_path=model_pat+'/srm/ckpt_best.pth'


spsl_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/spsl.yaml'
spsl_model_path=model_pat+'/spsl/ckpt_best.pth'


f3net_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/f3net.yaml'
f3net_model_path=model_pat+'/f3net/ckpt_best.pth'


ucf_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/ucf.yaml'
ucf_model_path=model_pat+'/ucf/ckpt_best.pth'

xception_detector_path='/Users/ahmedabdelaziz/Masters/Models/DeepfakeBench-main/training/config/detector/xception.yaml'
xception_model_path=model_pat+'/xception/ckpt_best.pth'

test_dataset=[ 'FF-NT','FF-F2F', 'FF-DF', 'FF-FS']




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
    for key in keys:
        filename = os.path.join("test-results", f"{key}_models.txt")

        with open(filename, "w") as file:
            file.write(f"Model details for key: {key}\n\n")

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
                header=model.__class__.__name__+' -- '+key
                with open(filename, "a") as file:
                    file.write(f"{header}\n  Error: {e} !!\n\n\n")
            else:
                y_prob_model = np.concatenate(model.prob)
                y_true = np.concatenate(model.label)
         
                fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob_model, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                acc= model.correct / model.total
                print('Model AUC:' , auc)
                print('Model Accuracy', model.correct / model.total)
                header=model.__class__.__name__+' -- '+key
                auc_str="AUC: ("+str(auc)+')'
                acc_str="Accuracy:  ("+str(acc)+')'
                with open(filename, "a") as file:
                    file.write(f"{header}\n{auc_str} -- {acc_str}\n\n\n")
    
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
         
    with open(efficientnetb4_detector_path, 'r') as f:
          efficientnetb4_config = yaml.safe_load(f)
    efficientnetb4_config['weights_path'] = efficientnetb4_model_path
    # # prepare the model (efficientnetb4_detector)
    efficientnetb4_model_class = DETECTOR[efficientnetb4_config['model_name']]
    efficientnetb4_model = efficientnetb4_model_class(efficientnetb4_config).to(cpu_device)
    efficientnetb4_ckpt = torch.load(efficientnetb4_model_path, map_location=cpu_device)
    efficientnetb4_model.load_state_dict(efficientnetb4_ckpt, strict=True)
    print('===> Load efficientnetb4 checkpoint done!')
         
    with open(f3net_detector_path, 'r') as f:
          f3net_config = yaml.safe_load(f)
    f3net_config['weights_path'] = f3net_model_path
    # # prepare the model (f3net_detector)
    f3net_model_class = DETECTOR[f3net_config['model_name']]
    f3net_model = f3net_model_class(f3net_config).to(cpu_device)
    f3net_ckpt = torch.load(f3net_model_path, map_location=cpu_device)
    f3net_model.load_state_dict(f3net_ckpt, strict=True)
    print('===> Load f3net checkpoint done!')
       
     
    with open(ffd_detector_path, 'r') as f:
          ffd_config = yaml.safe_load(f)
    ffd_config['weights_path'] = ffd_model_path
    # # prepare the model (FFD_detector)
    ffd_model_class = DETECTOR[ffd_config['model_name']]
    ffd_model = ffd_model_class(ffd_config).to(cpu_device)
    ffd_ckpt = torch.load(ffd_model_path, map_location=cpu_device)
    ffd_model.load_state_dict(ffd_ckpt, strict=True)
    print('===> Load FFD checkpoint done!')
         
    with open(fwa_detector_path, 'r') as f:
        fwa_config = yaml.safe_load(f)
    fwa_config['weights_path'] = fwa_model_path
    # # prepare the model (fwa_detector)
    fwa_model_class = DETECTOR[fwa_config['model_name']]
    fwa_model = fwa_model_class(fwa_config).to(cpu_device)
    fwa_ckpt = torch.load(fwa_model_path, map_location=cpu_device)
    fwa_model.load_state_dict(fwa_ckpt, strict=True)
    print('===> Load fwa checkpoint done!')

    with open(meso4_detector_path, 'r') as f:
        meso4_config = yaml.safe_load(f)
    meso4_config['weights_path'] = meso4_model_path
    # # prepare the model (meso4_detector)
    meso4_model_class = DETECTOR[meso4_config['model_name']]
    meso4_model = meso4_model_class(meso4_config).to(cpu_device)
    meso4_ckpt = torch.load(meso4_model_path, map_location=cpu_device)
    meso4_model.load_state_dict(meso4_ckpt, strict=True)
    print('===> Load meso4 checkpoint done!')

    with open(recce_detector_path, 'r') as f:
        meso4Inception_config = yaml.safe_load(f)
    meso4Inception_config['weights_path'] =meso4Inception_model_path
    #  # prepare the model (recce_detector)
    meso4Inception_model_class = DETECTOR[meso4Inception_config['model_name']]
    meso4Inception_model =meso4Inception_model_class(meso4Inception_config).to(cpu_device)
    meso4Inception_ckpt = torch.load(meso4Inception_model_path, map_location=cpu_device)
    meso4Inception_model.load_state_dict(meso4Inception_ckpt, strict=True)
    print('===> Loadmeso4Inception checkpoint done!')     

        
    with open(recce_detector_path, 'r') as f:
        recce_config = yaml.safe_load(f)
    recce_config['weights_path'] = recce_model_path
    # # prepare the model (recce_detector)
    recce_model_class = DETECTOR[recce_config['model_name']]
    recce_model = recce_model_class(recce_config).to(cpu_device)
    recce_ckpt = torch.load(recce_model_path, map_location=cpu_device)
    recce_model.load_state_dict(recce_ckpt, strict=True)
    print('===> Load recce checkpoint done!')
    
    with open(resnet34_detector_path, 'r') as f:
        resnet34_config = yaml.safe_load(f)
    resnet34_config['weights_path'] = resnet34_model_path
    # # prepare the model (resnet34_detector)
    resnet34_model_class = DETECTOR[resnet34_config['model_name']]
    resnet34_model = resnet34_model_class(resnet34_config).to(cpu_device)
    resnet34_ckpt = torch.load(resnet34_model_path, map_location=cpu_device)
    resnet34_model.load_state_dict(resnet34_ckpt, strict=True)
    print('===> Load resnet34 checkpoint done!')
          
    with open(spsl_detector_path, 'r') as f:
      spsl_config = yaml.safe_load(f)
    spsl_config['weights_path'] = spsl_model_path
    # # prepare the model (SPSL_detector)
    spsl_model_class = DETECTOR[spsl_config['model_name']]
    spsl_model = spsl_model_class(spsl_config).to(mps_device)
    spslckpt = torch.load(spsl_model_path, map_location=mps_device)
    spsl_model.load_state_dict(spslckpt, strict=True)
    print('===> Load SPSL checkpoint done!')
    
    with open(srm_detector_path, 'r') as f:
        srm_config = yaml.safe_load(f)
    srm_config['weights_path'] = srm_model_path
    # # prepare the model (srm_detector)
    srm_model_class = DETECTOR[srm_config['model_name']]
    srm_model = srm_model_class(srm_config).to(cpu_device)
    srm_ckpt = torch.load(srm_model_path, map_location=cpu_device)
    srm_model.load_state_dict(srm_ckpt, strict=True)
    print('===> Load srm checkpoint done!')
        
    with open(ucf_detector_path, 'r') as f:
        ucf_config = yaml.safe_load(f)
    ucf_config['weights_path'] = ucf_model_path
    # # prepare the model (UCF_detector)
    ucf_model_class = DETECTOR[ucf_config['model_name']]
    ucf_model = ucf_model_class(ucf_config).to(mps_device)
    ucf_ckpt = torch.load(ucf_model_path, map_location=mps_device)
    ucf_model.load_state_dict(ucf_ckpt, strict=True)
    print('===> Load UCF checkpoint done!')
        
    with open(xception_detector_path, 'r') as f:
        xception_config = yaml.safe_load(f)
    xception_config['weights_path'] = xception_model_path
    # # prepare the model (xceptionL_detector)
    xception_model_class = DETECTOR[xception_config['model_name']]
    xception_model = xception_model_class(xception_config).to(mps_device)
    xception_ckpt = torch.load(xception_model_path, map_location=mps_device)
    xception_model.load_state_dict(xception_ckpt, strict=True)
    print('===> Load xception checkpoint done!')

    



    
    
    
    

    
    


    models=[
        capsule_net_model,
            core_model,
            efficientnetb4_model,
            f3net_model,
            ffd_model,
            fwa_model,
            meso4_model,
            resnet34_model,
            spsl_model
            ,srm_model
            ,ucf_model,
            xception_model
            ]
    
    
    
    # # start training
    test_epoch(models, test_data_loaders)
  

if __name__ == '__main__':
    main()

import core.utils as utils
import pickle
import sys
from pathlib import Path

def check_models_accepts_modalities(model_architecture, model_task_names, modalities):
    for model_task_name in model_task_names:
        model_folder = utils.nnunet_model_folder / 'nnUNet' / model_architecture / model_task_name
        plans_path = list(model_folder.glob('**/plans.pkl'))
        if len(plans_path) > 0:
            with open(plans_path[0], 'rb') as f:
                data = pickle.load(f)
                message = f'The model {model_folder} takes the following modalities: {data["modalities"]}'
                print(message)
                for modality_name in data['modalities'].values():
                    if not any([modality in modality_name for modality in modalities]):
                        print(f'Error: the given modalities ({modalities}) do not match the model modalities {data["modalities"].values()}.')
                        # sys.exit(f'Error: the given modalities ({modalities}) do not match the model modalities.')
        else:
            print(f'Warning: the plans file of the model {model_folder} could not be found.')
    return

def predict(args, nnunet_task_folder, output_folder, model_task_name):

    print(f'Predictions will be output to {output_folder}.')
    
    print('predict...')

    command = ['nnUNet_predict', '-i', str(nnunet_task_folder / 'imagesTs'), '-o', str(output_folder), '-t', model_task_name, '-m', args.model_architecture, '-tr', 'nnUNetTrainerV2']
    if args.folds and 'None' not in args.folds:
        command += ['-f', ' '.join(args.folds)]
    
    if args.checkpoint:
        command += ['-chk', args.checkpoint]
    
    if utils.get_configuration('custom_nnunet_parameters', 'disable_tta'):
        command += ['--disable_tta']
    
    if 'custom_nnunet_parameters' in utils.configuration:
        command += ['--custom_parameters_path', args.configuration]

    utils.call(command, env=utils.environment_variables)

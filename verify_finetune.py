import os, sys, yaml, torch
sys.path.insert(0, '/home/y_rajeshwar/final_year_project_29/videomaev2_new/videomaev2_novel')

from train_classifier import main

if __name__ == '__main__':
    print('Testing fine-tuning initialization and checkpoint loading...')
    config_path = '/home/y_rajeshwar/final_year_project_29/videomaev2_new/videomaev2_novel/configs/config.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['training']['epochs'] = 1  # 1 epoch for verification
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
        
    main()

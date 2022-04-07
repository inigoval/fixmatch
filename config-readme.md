```type: 'fixmatch'                                      
project_name: 'ssl'                     # Project name for wandb logger
seed_i: 0                               # Initial seed
seed_f: 30                              # Final seed
data:
    n_labels: 2                         # Number of labels in data
    fraction: 1.0                       # Fraction of data-set to use for training
    split: 0.1                          # Fraction of data-set to use for labelled data
    val_frac: 0.1                       # Fraction of data to use for validation
    fri_R: -1                           # Artificially imbalance unlabelled data to given proportion of FRI samples (probably leave this). -1 does nothing.
    l: 'confident'                      # Data subset to use for labelled data [confident, uncertain, all]  
    u: 'rgz'                            # Data subset to use for unlabelled data [rgz, confident, uncertain, all]  

train:
    n_epochs: 800                       # Number of training epochs 

seed: null                              # Set seed (only needed for gridsearch)

###############################
### optimisable hyperparams ###
############################### 
n_aug: 2                                # Number of sequential strong augmentations to apply  
m_aug: 10                               # Magnitude of strong augmentations [0, 10]
cut_threshold: 23                       # Angular cut threshold for RGZ data
cutout: 0                               # Cutout size (not used if 0)
crop: False                             # Whether to use random crops
center_crop: 90                         # Center crop size
random_crop_min: 0.25                   # Random crop minimum
random_crop_max: 0.8                    # Random crop maximum
mu: 7                                   # Unlabelled batch size multiplier
lambda: 1                               # Weighting coefficinet for unlabelled loss term
tau: 0.95                               # Confidence threshold tau
p-strong: 1                             # Strong augmentation probability
lr: 0.0005                              # Learning rate
batch_size: 20                          # Batch size (scale this with labelled data volume)
```

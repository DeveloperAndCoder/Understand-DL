# Understand-DL
Project to understand how classifiers work under the hood and why they do what they do

## How to use

### Training
```
1. Train autoencoder - python3 autoencoder_STL.py run_number
2. Train classifier - python3 classifier_vgg.py run_number
3. Train combined Model - python3 combined_2_STL10.py -r run_number -a autoencoder/path -c classifier/path
```

### Testing
```
1. Accuracy calculation - python3 evaluate_single.py -r run_number -m model -mp model/path -d dataset
2. Plot graphs - python3 graphify.py
```

### Visualising
```
1. GRADCAM - python3 gradcam9.py -r run_number -m model/path
2. Diff - python3 diff_img.py -r run_number
3. Intermediate Results - python3 intermediate_output.py -r run_number
```

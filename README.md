# Keyphrase Generation: Lessons from a Reproducibility Study

Official code for the paper:   
Keyphrase Generation: Lessons from a Reproducibility Study  
Edwin Thomas and Sowmya Vajjala  
LREC-COLING 2024  

## Credits

The models used in the reproducibility study are heavily adapted from their respective original implementations:  
1. KPDrop: https://github.com/JRC1995/KPDrop  
2. UniKP: https://github.com/thinkwee/UniKeyphrase  
  
For Significance testing, the following repository was used: https://github.com/rtmdrr/testSignificanceNLP  

## Benchmarking Guidelines

### Dataset Preparation

Perform Tokenization according to Meng. et. al Deep Keyphrase Generation paper [KPG-OpenNMT-py](https://github.com/memray/OpenNMT-kpg-release)

Refer to `UniKP/OpenNMT-kpg-release/notebook/json_process.ipynb` for the tokenization step.

For final pre-processing of tokenized json files run the following script from `UniKP/UniKeyphrase/preprocess/start_make.sh`:

```
./start_make.sh
```

### Benchmarking UniKP

Navigate to `UniKP/UniKeyphrase/scripts` folder and run the following scripts:

### Train
```
# for PyTorch DDP based training
./start_train_ddp.sh
```

### Inference
```
./start_test.sh
```

### Compute AKP and PKP metrics
```
./metrics.sh
```

### Benchmarking KPDrop

Navigate to `KPDrop/scripts` folder and run the following scripts:

### Train
```
./train.sh
```

### Inference
```
./test.sh
```

## Significance Testing

Navigate to `SignificanceTesting/testSignificanceNLP` and run `automation.sh` with arguments for file A and file B should be changed to the generated metric files from previous benchmark steps. Please refer to `data_precessing.ipynb` for an example.


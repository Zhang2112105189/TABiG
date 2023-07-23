# TABiG: Time-Series Data Imputation via Realistic Masking-Guided Tri-Attention Bi-GRU
Author: **Zhipeng Zhang, Yiqun Zhang, An Zeng, Dan Pan, Yuzhu Ji, Zhipeng Zhang and Jing Lin**
# Run the code
To Run this code, you will first need to download the corresponding datasets and then perform the necessary data preprocessing using the code provided in the **Data_Processing** folder.

After data preprocessing, you can proceed with model training and testing using the **main_run.py**.

For example:
```bash
# Train:
python main_run.py --config_path Configs/AirQuality_Config.ini --dataset_name Air_Quality/datasets.h5 --model_name TABiG --cuda 0

# Test:
python main_run.py --config_path Configs/AirQuality_Config.ini --dataset_name Air_Quality/datasets.h5 --model_name TABiG --test_mode --test_step 2023-06-04_T09:32:33/model_trainStep_1897_valStep_271_imputationMAE_0.1472 --cuda 0
```

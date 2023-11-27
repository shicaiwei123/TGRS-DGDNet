# DGDNet
The code for Diversity-guided Distillation with Modality-center Regularization for Robust Multimodal Remote Sensing Image Classification

## Dependency
- Ubuntu20.04
- CUDA11.3
- PyToch1.12
- python3.8
- 
## Dataset
- Download
    - Original [Huston20013](https://github.com/danfenghong/ISPRS_S2FL), [Augsburg](https://github.com/danfenghong/ISPRS_S2FL)
    - Preprocessing [Huston2013](https://drive.google.com/drive/folders/1YSbAFzD9MKcNMBbYTeax_c1XNkSjZC_a) [Augsburg](https://drive.google.com/drive/folders/1f4bvCefoJ9Xd6QTbByDSBY5x7pAW1u2q)
    - precessing code: https://github.com/danfenghong/IEEE_TGRS_GCN/tree/master/DataGeneration_Functions
- Build soft link
  ```bash
  cd DGDNet
  mkdir data
  ln -s path_to_download_data ./data/dataset_name
  
  for example: ln -s /home/data/shicaiwei/remote_sensing/huston2013 ./data/huston2013
  ```

## Train

### Train process
- To average the results, for each sub-task, we train three models and choose the one with middle performance for the following task. 

### Train multimodal teacher
```bash
cd src
bash huston2013_F_HL_unimodal_center_patch.sh
bash huston2013_F_HM_share_unimodal_center.sh
bash augsburg_F_HSD_DGD.sh
```


### Distillation with DGD and MCR

```bash
cd src
bash huston2013_HM_transfer_share_unimodal_center.sh
bash huston2013_HM_transfer_share_unimodal_center.sh
bash augsburg_T_HSD_share_unimodal_center.sh
```

### Test
```bash
cd test
python multi_modal_share_unimodal_center.py 0 0+0 0 0 0
python augsburg_multimodal_unimodal_center.py 0 0+0 0 0 0
```


## Comparison Method
### Train
- Hemis
  ```bash
  cd src
   bash huston2013_F_HL_MV.sh
   bash huston2013_F_HM_MV.sh
   bash augsburg_F_HSD_hemis_tri.sh
  ```

- mmformer
  ```bash
  cd src
   bash huston2013_F_HL_mmformer.sh
   bash huston2013_F_HM_mmformer.sh
   bash augsburg_F_HSD_mmformer_tri.sh
  ```

- ShaSpec
  ```bash
  cd src
   bash huston2013_F_HL_ShaSpec.sh
   bash augsburg_F_HSD_shaspec_tri.sh
   bash huston2013_F_HM_ShaSpec.sh
  ```

### Test

  ```bash
    cd test
    python huston2013_hemis.py 0 0+0 0 0 0
    python huston2013_mmformer.py 0 0+0 0 0 0
    python huston2013_shaspec.py 0 0+0 0 0 0
  ```

## Abalation on Backbone
- ResNet
  ```bash
   cd src 
   bash huston2013_F_general_resnet_share_full.sh
   bash huston2013_T_general_RESNET_unimodal_share.sh
  ```
- AlexNet
  ```bash
    cd src
    bash huston2013_F_general_alex_share_full.sh
    bash huston2013_T_general_ALEX_share_unimodal.sh
  ```

- Test
  ```bash
    python multi_modal_general.py 0 0+0 0 0 0
  ```

## Visualization
- Prepairation
  - the patch of each pixel from the dataset
  - pretrained model
  - details can be seen in the function of **huston_prediction_plot** and **augsburg_prediction_plot** in prediction_plot.py  
- Code
    ```bash
    cd test
    python prediction_plot.py
    ```


# Image super resolution


## Related work:
- [A CSDN blog](https://blog.csdn.net/qianbin3200896/article/details/104181552?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164188419916780264030042%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164188419916780264030042&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-104181552.pc_search_insert_ulrmf&utm_term=%E8%B6%85%E5%88%86%E8%BE%A8%E7%8E%87%E9%87%8D%E5%BB%BA&spm=1018.2226.3001.4187) that describes details about what is super resolution
- [An excellent github](https://github.com/yjn870/SRCNN-pytorch), this repository was based on it, it offers me great help.
- [great job](https://github.com/xinntao/Real-ESRGAN)!!! strongly recommend
****
## Requirements(**install before running the code**):
- pytorch
- numpy
- h5py
  
****
## Use
**If you only want to use the model quickly, then directly jump to Step4**
### Step1:Prepare the dataset
- choice 1: download the standard dataset
  The 91-image(train set), Set5(test set) dataset converted to HDF5 can be downloaded from the links below.
  | Dataset | Scale | Type | Link |
  |---------|-------|------|------|
  | 91-image | 2 | Train | [Download](https://www.dropbox.com/s/2hsah93sxgegsry/91-image_x2.h5?dl=0) |
  | 91-image | 3 | Train | [Download](https://www.dropbox.com/s/curldmdf11iqakd/91-image_x3.h5?dl=0) |
  | 91-image | 4 | Train | [Download](https://www.dropbox.com/s/22afykv4amfxeio/91-image_x4.h5?dl=0) |
  | Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/r8qs6tp395hgh8g/Set5_x2.h5?dl=0) |
  | Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/58ywjac4te3kbqq/Set5_x3.h5?dl=0) |
  | Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/0rz86yn3nnrodlb/Set5_x4.h5?dl=0) |
  Download any one of 91-image and Set5 in the same Scale and then **move them under `./datasets` as `./datasets/91-image_x2.h5` and `./datasets/Set5_x2.h5`**
- choice 2:use your own pictures to make up of dataset
  ```python
  python prepare.py
  ```
### Step2:Train the model
```python
python train.py
```


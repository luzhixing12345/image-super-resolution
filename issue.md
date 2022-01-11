
# This file is used to record some errors
- OSError: Unable to open file (file locking disabled on this file system (use HDF5_USE_FILE_LOCKING<br>
  Solution:<br>
  ```shell
  #linux
  nano ~/.brashrc
  ```
  Add `export HDF5_USE_FILE_LOCKING='FALSE'` in a line, use `Ctrl-X` to exit
  ```shell
  source ~/.barshrc
  ```
- OSError: [WinError 1455] 页面文件太小，无法完成操作。 Error loading "C:\ProgramData\Anaconda3\envs\study_and_test\lib\site-packages\torch\lib\caffe2_detectron_ops_gpu.dll" or one of its dependencies.
  Solution:
  ```python
  python train.py --num-workers 0
  ```
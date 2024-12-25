# Machine Learning
This repository contains the code and resources for a machine learning project.

## 1. Install gpu for GeForce RTX 3060
##### Note: Please check compatible version, because it will be cost time, I had to spend 1 day to install them, so tired
1. Install Nvidia driver
2. Install CUDA toolkit = v11.7
3. Install Cudnn = 8.9.6
4. Add path environments
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\lib\x64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\libnvvp
C:\tools\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\bin
C:\tools\cudnn-windows-x86_64-8.9.6.50_cuda11-archive\include
```
6. Install requirement.txt file
7. Check GPU
```
python ./gpu_checking.py
```
8. Enjoy your GPU


### Bitwarden key code: MUDC 3V8F B50I WDYN 4MK1 B08Q C4FA YI2U
### Courses
1. Coursera
   - Andrew NG (perfect course)
2. VietAI
3. Youtobe MIT


## 2. Jupyter Notebook Server Setup
#### Step 1: Set a Password for Jupyter Notebook
```
jupyter notebook password
```

#### Step 2: Generate the Configuration File
```
jupyter notebook --generate-config
```

#### Step 3: Edit the Configuration File
```
c.NotebookApp.ip = '192.111.33.101'
c.NotebookApp.port = 8888  # You can change this port if needed
c.NotebookApp.open_browser = False
c.NotebookApp.allow_remote_access = True
```

#### Step 4: Start the Jupyter Notebook Server
```
jupyter notebook --config=C:\Users\Admin\.jupyter\jupyter_notebook_config.py
```

#### Accessing the Notebook
- Open your browser and navigate to `http://192.111.33.101:8888`.
- Use the password you set earlier to log in.
```
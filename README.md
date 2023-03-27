# RapidOCR_ONNX
This repository contain a hand on application for running RapidOCR detector with Onnxruntime models
___

### Content of repository

1. include - directory that contain hearders files for OCR  
2. models - ONNX models with featured dictionary of chars
3. source - directory that contain cpp files for OCR 
4. main.cpp - main file that represent the inference for the models


___
### Prerequisites

In order to build the application CMake and compiler is needed for linux could be used gcc.

Also this application need next libraries:

- OpenCV without or with CUDA
- ONNXRuntime with CUDA support or not


### GPU utilization
___
If the OpenCV and ONNXRuntime libraries were build with CUDA support - application expect to init GPUs but in case of GPU is not availiable or CUDA Toolkit is installed wrong - possible negative effect.



### Build
___
In order to build the application use CMakelists.txt [will be provided with next commit]
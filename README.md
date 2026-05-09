ParaGrad
Parallel Machine Learning Training using Distributed Gradient Descent with CPU multiprocessing and CUDA GPU acceleration.
Features
The project follows a modular folder structure to separate source code, datasets, notebooks, and experimental results. The src directory contains all model implementations and training scripts, including sequential and parallel Gradient Descent implementations. The notebook directory contains the experimental Jupyter Notebook used for training and visualization. The data directory stores raw and processed datasets, while the result directory stores generated plots and CSV result files.
project/
│
├── src/
│   ├── model.py
│   ├── sequential_gd.py
│   ├── sequential_gd_1.py
│   ├── parallel_gd.py
│   └── utils.py
│
├── notebook/
│   └── experiment.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── result/
│   ├── plots/
│   └── metrics/
│
└── README.md



CPU Sequential Training
CPU Sequential Training was implemented using Stochastic Gradient Descent (SGD) on a single CPU process. In this approach, all computations were performed sequentially without parallelism. This implementation was used as the baseline for evaluating the performance improvement achieved through multiprocessing and CUDA acceleration.

CPU Parallel Training
CPU Parallel Training was implemented using Python multiprocessing to simulate Distributed Gradient Descent. The dataset was divided into multiple partitions and distributed across several CPU processes. Each process independently computed gradients and updated model parameters. This approach improved execution speed by utilizing multiple CPU cores simultaneously.

CUDA Sequential Training
CUDA Sequential Training was implemented using CUDA-enabled PyTorch operations. The model and tensors were transferred to GPU memory, allowing matrix operations and neural network computations to execute on the GPU. Although the training workflow remained sequential, GPU hardware internally performed tensor computations in parallel using CUDA cores.

CUDA Parallel Training
CUDA Parallel Training combined parallel Gradient Descent with GPU acceleration. GPU-based tensor computation enabled large-scale parallel execution of matrix operations and gradient calculations. CUDA Parallel Training achieved the best overall performance among all implemented methods because of efficient GPU hardware parallelism and lower computational overhead.

Speedup & Efficiency Analysis
The project evaluated machine learning training performance using execution time, speedup, efficiency, and scalability metrics. Execution time was measured to compare training duration between sequential and parallel implementations. Speedup was used to evaluate performance improvement achieved through parallel execution, while efficiency measured resource utilization across multiple processes.

Datasets


MNIST
https://www.kaggle.com/datasets/hojjatk/mnist-dataset


Fashion-MNIST
https://www.kaggle.com/datasets/zalando-research/fashionmnist


Breast Cancer Wisconsin Dataset
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data


Technologies
Python, PyTorch, CUDA, Multiprocessing, Pandas, Matplotlib, Scikit-learn


Experimental Results

The experimental analysis demonstrated that parallel machine learning training significantly reduced execution time compared to sequential execution. CPU multiprocessing improved performance initially, but efficiency gradually decreased at higher process counts because of synchronization overhead and resource contention. CUDA-based GPU acceleration achieved significantly better scalability, higher speedup, and lower execution time compared to CPU multiprocessing. CUDA Parallel Training produced the best overall performance among all implemented training methods.

GPU Requirement

CUDA experiments require an NVIDIA CUDA-supported GPU and CUDA-compatible PyTorch installation. The project experiments were performed using an NVIDIA GeForce GTX 1650 GPU for CUDA acceleration and GPU-based parallel training.


Future Improvements

Future improvements may include implementing multi-GPU distributed training, larger benchmark datasets, deep convolutional neural network architectures, DistributedDataParallel (DDP), cloud GPU deployment, and advanced optimization algorithms. Additional scalability analysis using high-performance computing clusters can further improve distributed machine learning performance evaluation.

Author

Faria Soroni

CSE 706 Project — Parallel Algorithm

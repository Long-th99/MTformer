


## 1.(https://github.com/Long-th99/MTformer)MTformer: A Physics-Informed Spatio-temporal Transformer for Complex Dynamic System Modeling

## 2.Introduction
Accurately modeling complex spatiotemporal dynamics in systems such as fluid
dynamics and weather forecasting remains a grand challenge due to the difficulty in
balancing long-range dependency with numerical stability. Conventional CNN-based
and RNN-based frameworks exhibit limited capability in capturing global dependencies
and suffer from substantial error accumulation. While recent Transformer-based
models excel at capturing global correlations, their lack of explicit physical inductive
bias causes them to overfit statistical noise instead of underlying dynamical laws,
leading to the generation of non-physical high-frequency oscillations,termed "physical
artifacts",which undermines long-horizon forecasting stability.To address these issues,
we propose MTformer, a physics-guided model built upon a systematic interleaving
evolution framework. Our primary contribution is the strategic integration of TD-Blocks
and Fourier Neural Operators directly into the Gated Transformer backbone,
enabling a self-correcting feature evolution that alternates between neural-driven
attention and operator-driven physical constraints. In this architecture, the Gated
Transformer captures complex spatiotemporal nonlinearities, while the TD-Blocks
embed a Laplacian-type smoothness prior to mitigate non-physical drifts.
Simultaneously, the FNO components refine global spectral representations to
maintain multi-scale consistency.Extensive experiments demonstrate the effectiveness
of MTformer. On the TaxiBJ urban traffic prediction benchmark, the proposed model
achieves an MSE of 0.271 and an MAE of 14.31, outperforming all compared state-ofthe-art baselines. On the Navier–Stokes benchmark for strongly nonlinear flow
dynamics, MTformer also exhibits superior long-term forecasting accuracy and stability.
These results indicate that the proposed framework provides a robust  solution for
spatiotemporal prediction by jointly leveraging global attention, spectral modeling, and
Laplacian-inspired temporal regularization.Our code is available at
https://github.com/Long-th99/MTformer.

## 3.Flowchart
Here is a flowchart illustrating the overall process of the MTformer framework:
<img width="1794" height="1021" alt="Model" src="https://github.com/Long-th99/MTformer/raw/main/imgs/Model.png" />

## Quick Start
**1.Set uo the Enviroment**
Before running the application, you need to set up the environment. We recommend using a Python 3.8 virtual environment for better compatibility（Optional but Recommended).
Please execute the following commands：
```
python3.8 -m venv .venv
source .venv/bin/activate
```
**2.Install Required Dependencies**
Ensure you have **pip** installed, and then install all necessary Python packages by running the following command (all required libraries and corresponding versions are located in the requirements folder, or you can install them individually):
```
pip install -r requirements.txt # Install dependencies from requirements.txt
```
**3.Training Workflow**

<1>**Download Data:** To run the model on a specific dataset, you must download the corresponding dataset into the data folder. Create a subfolder within the data folder, name it using the **lowercase** name of the dataset, and place the dataset files inside this folder.

<2>**Configure Model Parameters:** Configure or modify the model parameters for the specified dataset within the configs folder to meet the dataset's training requirements.

<3>**Modify Time Steps:** Specific time steps, such as the prediction frames (e.g., 10, 20), can be modified in the configs files and in openstl/datasets/dataset_constant.py.

<4>**Modify Scripts:** You can modify the number of training epochs, the model, and other parameters in the corresponding files within the scripts folder. To resume training from a specific checkpoint, use --resume_from xxx, where xxx is the path to the model checkpoint from a previous run.

<5>**Execute Script:** Run the script (e.g., for the taxibj dataset, run ./scripts/taxibj/taxibj_MTFormer_train.sh).

## Contacts
If you have any questions about the code or the algorithm, please feel free to reach out to:  
**Email**: [longyu2005@qq.com](mailto:wym0152@163.com)

Note:
--The specified library versions must not be changed, otherwise, it may lead to execution failures.

--The provided data folder and work_dirs folder are currently empty. You will need to download the contents for the data folder yourself. We have also provided you with access to our public repository (containing the necessary data/files).
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM1MzY3NTk3OF19
-->

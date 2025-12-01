


## 1.(https://github.com/Long-th99/MTformer)MTformer: A Physics-Informed Spatio-temporal Transformer for Complex Dynamic System Modeling

## 2.Introduction
Deep learning modeling of spatio-temporal dynamic systems—such as fluid dynamics, weather forecasting, and traffic flow prediction—faces challenges in accurately capturing highly complex spatio-temporal strong dependencies and nonlinear evolutionary dynamics. Existing CNN- and RNN-based frameworks exhibit limitations in characterizing long-range dependencies.We propose a novel hybrid architecture, MTformer, which deeply integrates the global dependency capture mechanism of Transformers, the efficient frequency-domain modeling capability of Fourier Neural Operators (FNO), and a time-differenced physical diffusion module.The core innovation of MTformer lies in its use of a gated Transformer block (GTB) to efficiently capture spatiotemporal coupling relationships, while simultaneously employing a physical diffusion module to effectively suppress error accumulation. This significantly enhances the model's long-term prediction stability and physical interpretability. Furthermore, the introduced FNO mechanism further strengthens the model's ability to approximate solutions to complex nonlinear differential equations.Extensive experimental results demonstrate MTformer's superiority: on the urban dynamics prediction dataset TaxiBJ, the model achieves a mean squared error of 0.27 and a mean absolute error of 14.31, comprehensively outperforming all mainstream baseline models. On the Navier-Stokes dataset simulating strongly nonlinear systems, MTformer also exhibits outstanding long-term inference performance.These findings indicate that MTformer achieves a significant leap in prediction accuracy through the organic coupling of the Transformer backbone with physical time-differencing and FNO, making it particularly suitable for precise modeling and long-term forecasting of strongly nonlinear dynamic systems.

## 3.Flowchart
Here is a flowchart illustrating the overall process of the MTformer framework:
![](/imgs/2025-12-01/7db2mkyC7Fzc6Odx.png)
<img width="1794" height="1021" alt="Model" src="https://github.com/user-attachments/assets/f572df30-9d03-40ca-8297-aab5a7a40d8a" />


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
1.**Download Data:** To run the model on a specific dataset, you must download the corresponding dataset into the data folder. Create a subfolder within the data folder, name it using the **lowercase** name of the dataset, and place the dataset files inside this folder.
2.**Configure Model Parameters:** Configure or modify the model parameters for the specified dataset within the configs folder to meet the dataset's training requirements.
3.**Modify Time Steps:** Specific time steps, such as the prediction frames (e.g., 10, 20), can be modified in the configs files and in openstl/datasets/dataset_constant.py.
4.**Modify Scripts:** You can modify the number of training epochs, the model, and other parameters in the corresponding files within the scripts folder. To resume training from a specific checkpoint, use --resume_from xxx, where xxx is the path to the model checkpoint from a previous run.
5.**Execute Script:** Run the script (e.g., for the taxibj dataset, run ./scripts/taxibj/taxibj_MTFormer_train.sh).

## Contacts
If you have any questions about the code or the algorithm, please feel free to reach out to:  
**Email**: [longyu2005@qq.com](mailto:wym0152@163.com)

Note:
--The specified library versions must not be changed, otherwise, it may lead to execution failures.
--The provided data folder and work_dirs folder are currently empty. You will need to download the contents for the data folder yourself. We have also provided you with access to our public repository (containing the necessary data/files).
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM1MzY3NTk3OF19
-->

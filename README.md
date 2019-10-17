Algorithmic differentiation (algodiff)
======================================

This repository contains code and data to generate OpenSim-based trajectory optimization of human movement using algorithmic differentiation (AD) as described in: Falisse A, Serrancoli G, Dembia C, Gillis J, De Groote F, (2019). Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement. PLOS ONE.

We enabled the use of AD in OpenSim through a custom source code transformation tool named Recorder and through the operator overloading tool ADOL-C. To enable the use of AD, we modified the source code of OpenSim and Simbody: https://github.com/antoinefalisse/opensim-core/tree/AD-recorder.

We then developed an interface between OpenSim and [CasADi](https://web.casadi.org/)) to solve trajectory optimization problems. CasADi is a tool for nonlinear optimization and algorithmic differentiation. 

We compared the computational efficiency of using standard finite differences (FD) versus AD (through Recorder (AD-Recorder) and ADOL-C (AD-ADOLC)) when solving a series of trajectory optimization problems. These problems consisted of simulations of perturbed balance, two-dimensional predictive simulations of walking, and three-dimensional tracking simulations of walking. We found that using AD through Recorder was between 1.8 ± 0.1 and 17.8 ± 4.9 times faster than using FD, and between 3.6 ± 0.3 and 12.3 ± 1.3 times faster than using AD through ADOL-C. The more complex the model / problem the larger the benefit of using AD instead of FD.

In this repository, we provide code necessary to perform these simulations with AD-Recorder and FD (providing OpenSim and Simbody modified for use with ADOL-C is more complicated and less relevant since Recorder is more efficient than ADOL-C as well as easier to use). We also provide code for analyzing the results and reproduce the figures presented in the associated paper.

Brief overview of the repository
--------------------------------

* Predictive simulations of walking with a 2D musculoskeletal model
    * The main script is `PredSim_2D.m` in the folder `predictiveSimulations_2D/OCP`. Running this script will formulate and solve the trajectory optimization problem / optimal control problem (OCP) underlying a predictive simulation of walking. This script calls several functions saved in the other folders of `predictiveSimulations_2D`. The best way to get started is therefore to run this script and look at the different steps. When the optimization has converged, the results will be saved in the folder `predictiveSimulations_2D/Results/PredSim_2D`. Among the saved results is a motion file that can be loaded in the OpenSim GUI using the musculoskeletal model saved as `predictiveSimulations_2D/OpenSimModel/subject1/subject1_scaled.osim`.
    
* Tracking simulation of walking with a 3D musculoskeletal model
    * The main script is `TrackSim_3D.m` in the folder `trackingSimulations_3D/OCP`. Running this script will solve the trajectory optimization problem / optimal control problem (OCP) underlying a tracking simulation of walking. This script calls several functions saved in the other folders of `trackingSimulations_3D`. The best way to get started is therefore to run this script and look at the different steps. When the optimization has converged, the results will be saved in the folder `trackingSimulations_3D/Results/PredSim_2D`. Among the saved results is a mot file that can be loaded in the OpenSim GUI using the musculoskeletal model saved as `trackingSimulations_3D/OpenSimModel/subject1/subject1.osim`.
    
* Simulation of perturbed balance with inverted pendulums of varying complexities (between 2 and 10 degrees of freedom)
    * The main script are `OCP_pendulum_<#>dofs` in the folders `pendulumSimulations/Pendulum_<#>dofs`. Note that we called those scripts from `batch_OCP_pendulum_<#>dofs` to run the many different cases investigated in this study.
    
* Scripts for analyzing results
    * The scripts in the folder `analysisResultsForPaper` enable analyzing the results and reproducing the figures presented in the paper.
    
Brief overview of the framework
--------------------------------

A detailed description of our framework to solve trajectory optimization problems with OpenSim is provided in Figure 2 of the paper. In brief, our framework enables making an OpenSim function `F` its derivatives available within the CasADi environment for use the NLP solver during an optimization. In more detail, the function `F` is first described as C++ code (e.g., `predictiveSimulations_2D/ExternalFunctions/PredSim_2D.cpp`). We then use the Recorder tool to provide the expression graph of `F` as MATLAB source code. From this MATLAB code, we can use CasADi’s C-code generator to generate C-code containing `F` and expressions for its derivatives. This C-code can then be compiled as a Dynamic-link Library (DLL) that can be imported as an external function within the CasADi environment. In our application, `F` represents the multi-body dynamics and is called when formulating the optimal control problem. The latter is then composed into a differentiable optimal control transcription using CasADi. During the optimization, CasADi provides the NLP solver with evaluations of the NLP objective function, constraints, objective function gradient, constraint Jacobian, and Hessian of the Lagrangian. You can find more information about how to build this pipeline in this repository: https://github.com/antoinefalisse/opensim-core/tree/AD-recorder.
    
Thanks for citing our work in any derived publication. Feel free to reach us for any questions: antoine.falisse@kuleuven.be | antoinefalisse@gmail.com | friedl.degroote@kuleuven.be | gil.serrancoli@upc.edu. This code has been developed on Windows using MATLAB2017b. There is no guarantee that it runs smooth on other platforms. Please let us know if you run into troubles.

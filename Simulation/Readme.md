### Monte-Carlo Simulation

This directory holds all necessary data to reproduce dose-simulations based on the CBCT data provided in the Slice2Volume repository on [RODARE](https://rodare.hzdr.de/deposit/810). 

The simulations can be conducted with the TOPAS software with the SpaceFiles under MonteCarlo_InputFiles. The output can be padded to the dimensions of the CBCT data with the script zeropadding.py. The referred tables for the conversion of CT numbers (Hounsfield units) to Materials are supplied in thhe following publications:

* Permatasari, F. F., Eulitz, J., Richter, C., Wohlfahrt, P., & Lühr, A. (2020). Material assignment for proton range prediction in Monte Carlo patient simulations using stopping-power datasets. Iopscience.Iop.Org. https://doi.org/10.1088/1361-6560/ab9702
* Schneider, W., Bortfeld, T., & Schlegel, W. (2000). Correlation between CT numbers and tissue parameters needed for Monte Carlo simulations of clinical dose distributions. Physics in Medicine and Biology, 45(2), 459–478. https://doi.org/10.1088/0031-9155/45/2/314

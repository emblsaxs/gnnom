# Introduction
Small-angle X-ray scattering (SAXS) experiments are widely used for the characterization of biological macromolecules in
solution. SAXS patterns contain information on the size and shape of dissolved particles in nanometer resolution. Here
we propose a novel method for primary SAXS data analysis based on the application of artificial neural networks (NN).
Trained on synthetic SAXS data, the feedforward neural networks are able to reliably predict molecular weight and
maximum intraparticle distance (Dmax) directly from the experimental data. The method is applicable to data from
monodisperse solutions of folded proteins, intrinsically disordered proteins, and nucleic acids. Extensive tests on
synthetic SAXS data generated in various angular ranges with varying levels of noise demonstrated a higher accuracy and
better robustness of the NN approach compared to the existing methods. 

# How to apply NN
To apply the NN you need to use e.g. the following command

```bash
python apply_nn.py p mw /path/to/my/datafile 1.0 2.8 --units=nanometer --n=10000
```
Here **p** stands for type of particle (proteins); **mw** - what parameter to predict (mw or dmax); **1.0** and **2.8** are the I(0) and Rg, accordingly; **units** are the angular units (angstrom or nanometer) and **n** is the number of repetitions for resampling.
The command 
```bash
python apply_nn.py -h
```
will print the help menu and exit.

# How to train a new NN
To train a NN you need the training, validation and test set. The simulated concentrations used for training are hard-coded into the _makemodel_scalar.py_ script as the "folders" variable. To run the script you can use e.g. the following command

```bash
python makemodel_scalar.py /path/to/data /path/to/crysol/logfiles 500 mw 
```
Here **/path/to/data** implies the structure similar to [this one](https://oc.embl.de/index.php/s/fdisAFWzws0nkW9). Then goes **/path/to/crysol/logfiles** - the path to the CRYSOL log files; **500** - number of epochs; and **mw** is the desired parameter. 
It is also possible to re-use old weights if you want to train your model up (*.h5 file), or to use a pickle file to quickly re-read the input data.  
The command 
```bash
python makemodel_scalar.py -h
``` 
will print the help menu and exit.

# Training/validation/test sets
The original data sets generated from PDB that were used for training NN and benchmarking it against other methods are available [here](https://oc.embl.de/index.php/s/fdisAFWzws0nkW9)

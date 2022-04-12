# Introduction
Small-angle X-ray scattering (SAXS) experiments are widely used for the characterization of biological macromolecules in
solution. SAXS patterns contain information on the size and shape of dissolved particles in nanometer resolution. Here
we propose a novel method for primary SAXS data analysis based on the application of artificial neural networks (NN).
Trained on synthetic SAXS data, the feedforward neural networks are able to reliably predict molecular weight and
maximum intraparticle distance (Dmax) directly from the experimental data. The method is applicable to data from
monodisperse solutions of folded proteins, intrinsically disordered proteins, and nucleic acids. Extensive tests on
synthetic SAXS data generated in various angular ranges with varying levels of noise demonstrated a higher accuracy and
better robustness of the NN approach compared to the existing methods. 

![alt text](https://github.com/emblsaxs/gnnom/blob/master/header.jpg?raw=true)

# How to cite
If you use gnnom for your research please cite:
D.Molodenskiy, D.Svergun, A.Kikhney (2022) Artificial neural networks for solution scattering data analysis, Structure 
[DOI](https://doi.org/10.1016/j.str.2022.03.011)

# How to apply NN
To apply the NN you need to use a command similar to this:

```bash
python apply_nn.py p mw /path/to/my/datafile 1.0 2.8 --units=nanometer --n=10000
```
Here **p** stands for type of particles (proteins); **mw** - what parameter you want to predict (mw or dmax); **1.0** and **2.8** are the I(0) and Rg values, accordingly; **units** are the angular units (inverse angstroms or inverse nanometers) and **n** is the number of repetitions for resampling. The latter parameter is usually set to about 10 000.
The following command 
```bash
python apply_nn.py -h
```
will print the help menu and exit.

# How to train a new NN
To train a NN you need the training, validation and test sets. To run the script you can use e.g. the following command

```bash
python makemodel_scalar.py /path/to/data /path/to/crysol/logfiles 500 mw 
```
Here **/path/to/data** implies the structure similar to [this one](https://oc.embl.de/index.php/s/fdisAFWzws0nkW9). The root directory should contain the subfolders with different simulated concentrations used for training and the list of these folders is hard-coded into the _makemodel_scalar.py_ script as the "folders" variable. If you are using your own training set, please change this variable in the code. 
The next parameter is **/path/to/crysol/logfiles** - the path to the CRYSOL log files; **500** is the number of epochs for training; and **mw** is the desired parameter. 
It is also possible to re-use old weights if you want to train your model up using previously generated *.h5 file. You can also use a pickle file to quickly re-read the input data.  
The command 
```bash
python makemodel_scalar.py -h
``` 
will print the help menu and exit.

# Training/validation/test sets
The original data sets were generated from [PDB](https://www.rcsb.org/), [PED](https://proteinensemble.org/) and [NDB](http://ndbserver.rutgers.edu/) databases that were used for training of the NNs and benchmarking it against other methods. The data are available [here](https://oc.embl.de/index.php/s/fdisAFWzws0nkW9)

# Web service
If you don't want to modify the code and just want to try it out as is on your SAXS data, this approach is also implemented as a [web service](https://dara.embl-hamburg.de/mwdmax.php). Please make sure that you select the proper macromolecule type and (in case you are using WAXS data) the angular units!

# RAGNAR
![](cover_picture.jpeg)

```RAGNAR```, the Resolution Adaptive Generator of Nocturnal Airglow Radiation, to generate robust line lists for wavelength calibration and climatology.

```RAGNAR``` works by simulating the emission of the nocturnal airglow in the optical and near-infrared, varying all relevant parameters to determine the intrinsic variability of each possible line, at a given resolution. A quick overlook of the philosphy of the inner workings of the code is

```
for Condtion, Instrumental Effect in {Conditions, Instruments}:
    Simulate spectrum
    Find peak wavelengths
    Save peaks

For peaks in saved peaks:
    Crossmatch stable peaks
    Crossmatch unstable peaks
    Find mean and variation of the wavelength and intensity of all peaks

For wavelength in mean wavelengths:
    Retrieve extra line information (transmission, closest neighbor, labels, ...)
    Make line list
```

The **installation** is simple - just clone the code in this repository, and download the transmission curves at https://drive.google.com/drive/folders/17kQAoU3txWDdn7Z1USVlhdDSTdPjHE_i?usp=sharing, and modify the ```base``` keyword to be your local path to the downloaded transmission curves. 

```RAGNAR``` is very flexible, and takes a long range of keywords


- `min\_wav = 360`: Minimum of the wavelength range to consider [nm].
- `max\_wav = 1300`: Minimum of the wavelength range to consider [nm].
- `steps\_per\_line = 100`: Number of steps per line spread interval for the wavelength grid over which calculations are done. Turning this up will lead to a slower but more precise calculation. It will especially yield variability information for smaller variabilities.
- `line\_spread = 0.1`: The Gaussian line width of lines, as measured by the spectrograph [nm]. If `resolution` is set, `line\_spread` will be disregarded.
- `resolution = $\infty$`: The resolution of the spectrograph, as. The default is $\infty$, since infinite resolution added with a non-zero `line\_spread`, will not change the `line\_spread`. If the resolution is not $\infty$ or `None`, `line\_spread` will be automatically set to 0.
- `base = ~/../../tigress/cj1223/`: The base path to the `MODTRAN` high-resolution outputs.
- `originDf\_file = grid\_mk50`: The additional path to the `MODTRAN` low-resolution curve to use for determining which species are likely to be the cause of absorption.
- `Trots = [175, 190, 205]`: Array or list of the rotational temperatures [K].
- `Tvibs = [8000, 10500, 13000]`: Arrays or lists of the vibrational temperatures for the OH [K].
- `save\_name = 'test'`: If `None`, do not save; else, save line list in the `lineLists` directory, which will be created automatically if it does not exist. The file will be saved in .csv and .pkl formats. The filename will have the date added.
- `importance\_limit\_group = 2e-2`: If a subline contributes less than this fraction of the total intensity at line centre, do not include it in the label. However, all lines that contribute down to `importance\_limit\_group$^3$` are included in `originIds` column. This makes precision analysis of each line possible if needed.
- `chem\_scatters = 3`: The number of samples from the species concentration distribution to take.
- `dlambda = 15`: How far to either side of the centre of a given line to go to search for other lines which may impact that line, in units of the Gaussian line width.
- `intensity\_cut = 0`: Consider only peaks that are stronger than this as relevant lines for the line list. The default is 0, such that the cut on the intensity cut can be done later.
- `inclusion\_limit = 1e-3`: The limit for how much a line should contribute at its line centre for us to include it in calculations. The default is 1/1000, so any line which contributes less than 1/1000th of the total intensity at its line centre will not be carried forward. Increasing this will let the code run more quickly, but it should not be increased beyond 1/100.
- `linewidth\_scatters = 1`: If 1, take zero uncertainty in the LSF and run only at the fiducial resolution; otherwise, do `linewidth\_scatters` uniform samples of the LSF error distribution.
- `LSF\_err = 0.03`: The width of the Gaussian from which to sample LSF perturbations. `LSF\_err`$\cdot100$ is the representative percentual variabilility in the width of the LSF.
- `continuum = True`: Whether or not to add in the continuum. This is computationally inexpensive so there's no reason to leave this out.
- `aerosols = ['background', 'volcanic']`: List of aerosol models to loop over for the transmission curves. 'background', and 'volcanic' are possible options.
- `pwvs = [0.0, 2.0]`: List precipitable water vapours (PWV) to loop over for the transmission curves [mm].
- `zds = [0, 40]`: List of zenith declination of the telescope to loop over for the transmission curves [degrees].
- `use\_centroider = False`: Whether or not to use a centroider to find the peaks of the spectra. The centroider convolves the spectrum with the LSF before finding maximum likelihood peaks.
- `total\_lim = 0.95`: If a line has transmission below this, consider it significantly attenuated.
- `contribution\_lim = 0.99`: If a species has transmission below this for a given line, it contributes significantly.
- `important\_species = ['combin', 'aercld', 'molec', 'H2O', 'O3', 'O2', 'CO2', 'CH4']`: Possible absorption origins to consider for determining which species cause absorption. See below in `outputs` for a more detailed description of each species.
- `verbose = 1`: 0 = print almost nothing, 1 = print enough to follow progress, 2 = print enough to do some approximate debugging.

For all of these fun options, sophisticated line lists are output, containing information about

- `wav`: The mean wavelength of the line.
- `sigma\_wav`: The standard deviation of the wavelength of the line, added in quadrature with the numerical error in the simulation.
- `separation`: The distance to the nearest line in wavelength space.
- `intensity`: The mean intensity of the line at line centre, given in kilorayleigh.
- `sigma\_intensity`: The standard deviation of the line intensity at line center.
- `E\_A\_eff`: The effective Einstein A - coefficient, calculated as the intensity-weighted average of the $A$s of the individual sublines.
- `continuum`: The approximate continuum at line center.
- `label`: The label of the line. If there are multiple lines, the label will consist of the label of each subline separated by $|$.
- `number\_of\_lines`: The total number of sublines that make up the line.
- `clean`: `True` if all of the sublines originate from the same upper state and therefore vary exactly together. `False` otherwise. Since $\Lambda$-doublets vary together, these are not taken into account when determining if the line is clean. For studying the dynamics of a given system, only the lines marked as `clean` should be used. 
- `robust`: `True` if the line is identifiable across all simulations. `False` otherwise. These lines are included mainly such that a line that may be visible in a given exposure can be identified although it is not part of the line list used for calibration.
- `detection\_prob`: The fraction of the simulated spectra that the line is identifiable in.
- `subline\_wav`: The wavelengths of the sublines that contribute more than a preset fraction of the total line intensity at line centre. 
- `subline\_I`: The fractional intensity of the line that the sublines contribute at line center. The sublines must contribute more than a preset fraction of the total line intensity at line center. 
- `err\_important`: The intensity error in considering only the sublines which contribute more than a preset fraction of the total line intensity at line centre.
- `origin\_Ids`: The `ID`'s of the lines in the table which contain detailed information about all sublines.
- `transmission`: The average atmospheric transmission of the line.
- `transmission\_origin`: A bit-mask to identify the most likely origin of the absorption. The considered species are 
    - `'combin',` the total transmission. 1 if the transmission is below 0.95,
    - `'aercld',` aerosols, 1 if the transmission due to this species is below 0.99,
    - `'molec',` Rayleigh scattering, 1 if the transmission due to this species is below 0.99,
    - `'H2O',` water vapour, 1 if the transmission due to this species is below 0.99,
    - `'O3',` ozone, 1 if the transmission due to this species is below 0.99,
    - `'O2',` molecular oxygen, 1 if the transmission due to this species is below 0.99,
    - `'CO2',` carbon dioxide, 1 if the transmission due to this species is below 0.99,
    - `'CH4'` Methane, 1 if the transmission due to this species is below 0.99.

As an example, a line with a total transmission of 0.92, due to aerosols and oxygen would therefore be marked `11000100`.
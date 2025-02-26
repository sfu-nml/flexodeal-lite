# Flexodeal Lite

**Check the [release notes](https://github.com/sfu-nml/flexodeal-lite/releases) to learn more about the new features of the latest version!**

This is a reduced version of the [Flexodeal](https://github.com/javieralmonacid/flexodeal) library which has been structured to follow [deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html) guidelines. It is intended to simulate the deformation of a block of muscle tissue (similar to a biopsy). The stress response has been simplified to include contributions from muscle fibres (Hill-type model) and the base material (hyperelastic, Yeoh).

Check out this [cool video](https://youtu.be/CCTiSV1Vl7o) of a fully dynamic, fully active, isometric contraction!

This is a library written in C++ meant to be run in Linux and MacOS environments. Windows users can use this library via WSL (Windows Subsystem for Linux). A graphical interface is not yet available.

## Brief description of Flexodeal

This code is used to model skeletal muscle in both dynamic and quasi-static experiments, and can include contributions from muscle, aponeurosis, fat, and tendon material. The underlying material properties, numerical algorithms, are described in the following paper:

* Almonacid, J. A., Domínguez-Rivera, S. A., Konno, R. N., Nigam, N., Ross, S. A., Tam, C., & Wakeling, J. M. (2024). A three-dimensional model of skeletal muscle tissues. SIAM Journal on Applied Mathematics, S538-S566. [https://doi.org/10.1137/22M1506985](https://doi.org/10.1137/22M1506985)

This coding framework has been used in the following studies:

* Ross, S. A., Domínguez, S., Nigam, N., & Wakeling, J. M. (2021). The Energy of Muscle Contraction. III. Kinetic Energy During Cyclic Contractions. Frontiers in Physiology, 12(April), 1–16. https://doi.org/10.3389/fphys.2021.628819

* Konno, R. N., Nigam, N., & Wakeling, J. M. (2021). Modelling extracellular matrix and cellular contributions to whole muscle mechanics. PLoS ONE, 16(4 April 2021), 1–20. https://doi.org/10.1371/journal.pone.0249601

* Ryan, D. S., Domínguez, S., Ross, S. A., Nigam, N., & Wakeling, J. M. (2020). The Energy of Muscle Contraction. II. Transverse Compression and Work. Frontiers in Physiology, 11(November), 1–15. https://doi.org/10.3389/fphys.2020.538522

* Wakeling, J. M., Ross, S. A., Ryan, D. S., Bolsterlee, B., Konno, R., Domínguez, S., & Nigam, N. (2020). The Energy of Muscle Contraction. I. Tissue Force and Deformation During Fixed-End Contractions. Frontiers in Physiology, 11, 1–42. https://doi.org/10.3389/fphys.2020.00813

* Domı́nguez S. From eigenbeauty to large-deformation horror. Ph.D. Thesis, Simon Fraser University. 2020. Available from: http://summit.sfu.ca/item/20968

* Ross, S. A., Ryan, D. S., Dominguez, S., Nigam, N., & Wakeling, J. M. (2018). Size, history-dependent, activation and three-dimensional effects on the work and power produced during cyclic muscle contractions. Integrative and Comparative Biology, 58(2), 232–250. https://doi.org/10.1093/icb/icy021

* Rahemi, H., Nigam, N., & Wakeling, J. M. (2015). The effect of intramuscular fat on skeletal muscle mechanics: implications for the elderly and obese. Journal of The Royal Society Interface, 12(109), 20150365. [https://doi.org/10.1098/rsif.2015.0365](https://doi.org/10.1098/rsif.2015.0365)

* Rahemi, H., Nigam, N., & Wakeling, J. M. (2014). Regionalizing muscle activity causes changes to the magnitude and direction of the force from whole muscles—a modeling study. Frontiers in physiology, 5, 298. [https://doi.org/10.3389/fphys.2014.00298](https://doi.org/10.3389/fphys.2014.00298)


## How to use it?

- Make sure you have properly set up deal.II v9.3 (or newer) and its dependencies. For more information, visit [their website](http://www.dealii.org).
- Get the latest release [here](https://github.com/sfu-nml/flexodeal-lite/releases). Alternatively, you can clone the repository for bleeding edge updates: ```git clone https://github.com/sfu-nml/flexodeal-lite.git```.
- Navigate to the Flexodeal folder using ```cd```.
- Call CMake:
    - In release mode: ```cmake . -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=<path/to/deal.II>```.
    - In debug mode: ```cmake . -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=<path/to/deal.II>```.
- Call make. A simple call to ```make``` should suffice.
- Run the code. This either achieved by calling ``` make run``` or ```./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation.dat ```. A folder with the current timestamp will be created. This is where the results of your execution will be stored.

## How long does it take to run?

In an Intel i5-9600K (3.70 GHz x 6) machine the code with its default settings (dynamic) takes about 4 minutes to complete 500 ms of simulation. In turn, the quasi-static code (`set Type of simulation = quasi-static`) takes only 18 seconds! That speaks volumes about the nonlinearity of the dynamic problem.

## Binary files: what they are and how to read them

A binary file is a file that contains data in a format that is not directly readable by humans. Unlike text files, which store data as plain text (ASCII or Unicode), binary files store data in raw binary format, which is optimized for computer processing rather than human readability.

Each of these files (which are written at each time step) contains a two-dimensional array (number of quadrature points) x (number of columns),
where each column represents a different quantity of interest and each row is a different quadrature point.

Since these are rather large files, these are **not** output by default. To export these files, set in `parameters.prm`:
```
subsection Output binary files
    set Output binary files main variables = true
    set Output binary files tensors = true
end
```

The first type of files exports binary files named `cell_data_main-3d-XYZ.data`, where XYZ is the time step. Each one of these files contains the following columns (20 in total):
1. `qp_x` (X component of the quadrature point in the reference configuration)
2. `qp_y` (Y component of the quadrature point in the reference configuration)
3. `qp_z` (Z component of the quadrature point in the reference configuration)
4. `JxW` (dilation J x quadrature point weight)
5. `det_F` (determinant of the deformation tensor)
6. `u1` (First component of current displacement)
7. `u2` (Second component of current displacement)
8. `u3` (Third component of current displacement)
9. `v1` (First component of current velocity)
10. `v2` (Second component of current velocity)
11. `v3` (Third component of current velocity)
12. `p` (Current pressure)
13. `D` (Current dilation)
14. `stretch` (Fibre stretch)
15. `stretch_bar` (Fibre stretch, isochoric component)
16. `strain_rate` (Fibre strain rate, normalized)
17. `strain_rate_bar` (Fibre strain rate, normalized, isochoric component)
18. `orientation_x` (X component of current fibre orientation)
19. `orientation_y` (Y component of current fibre orientation)
20. `orientation_z` (Z component of current fibre orientation)

The second type of files are named `cell_data_tensors-3d-XYZ.data` and contains the following columns (see the comments in the parameter file to view the exact name of each column, 68 columns in total):
- Column 1: `qp_x`
- Column 2: `qp_y`
- Column 3: `qp_z`
- Column 4: `Jxw`
- Column 5: `det_F`
- Columns 6-14: `F_i_j`, the (i,j) component of the deformation tensor `F`
- Columns 15-23: `tau_i_j`, the (i,j) component of the Kirchhoff stress `tau`
- Columns 24-32: `tau_vol_i_j`, the (i,j) component of `tau_vol`
- Columns 33-41: `tau_iso_i_j`, the (i,j) component of `tau_iso`
- Columns 42-50: `tau_muscle_active_i_j`, the (i,j) component of `tau_muscle_active`
- Columns 51-59: `tau_muscle_passive_i_j`, the (i,j) component of `tau_muscle_active`
- Columns 60-68: `tau_muscle_base_i_j`, the (i,j) component of `tau_muscle_base`

To read these files in Matlab, use the following function:
```
function df = read_binary(filename, ncols)
% READ_BINARY Transforms a the contents of a binary file into a matrix
%
% df = read_binary(filename, cols) reads the file "filename" assuming that
% it will find a vector of "float32" numbers. Assuming this vector has N
% elements, the array is then reshaped to a N/ncols x ncols matrix, where
% ncols is the number of columns the array is expected to have.

fid = fopen(filename);
df = fread(fid,Inf,"float32");
df = reshape(df', ncols, length(df)/ncols)';
fclose(fid);
```

## Latest line count (in main branch)

```
Tue 25 Feb 2025 07:03:51 PM PST

>> cloc --exclude-dir=.vscode --exclude-lang=JSON,XML,make .
    
    10 text files.
    10 unique files.                              
    8 files ignored.

github.com/AlDanial/cloc v 1.90  T=0.02 s (181.2 files/s, 323568.2 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              1            720           1127           3329
Markdown                         1             36              0            105
CMake                            1              6             15             18
-------------------------------------------------------------------------------
SUM:                             3            762           1142           3452
-------------------------------------------------------------------------------

```

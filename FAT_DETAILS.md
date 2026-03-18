# Implementation details

## Parameter files

- In `parameters.prm`, add a new section
```
# ---------------------- Fat properties ------------------------

# Bulk modulus fat [Pa]
set Bulk modulus fat = 1.0e+07

# Fat "fudge" factor [nondimensional].
# Ideally this should be 1.
set Fat factor = 1.0

# Constants in Neo-Hookean strain-energy function
set Fat constant 1 = 1.3e+05

# Fat fraction
set Fat fraction = 0.0
```
## Main file

In `flexodeal.cc`:

1. Modify the `MuscleProperties` struct. Add four new variables, including declarers and parsers.

2. In `Muscle_Tissues_Three_Field`: 
    - add 4 new inputs to the constructor, three of them become new class members. 
    - The input `kappa_fat` is used uniquely to homogenize the value of `kappa_muscle` and is not defined as a new class member.
    - `get_tau_bar()` needs to be homogenized in the same way. This requires to define `get_tau_fat_bar()`
    - `get_c_bar()` also needs to be homogenized. This requires to define `get_c_fat_bar()`

3. In `PointHistory::setup_lqp()`, modify the material variable according to the previous modifications to the constructor


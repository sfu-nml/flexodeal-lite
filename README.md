# Flexodeal Lite
This is a reduced version of the [Flexodeal](https://github.com/javieralmonacid/flexodeal) library which has been structured to follow [deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html) guidelines. 

It is intended to simulate the deformation of a block of muscle tissue (similar to a biopsy). The stress response has been simplified to include contributions from muscle fibres (Hill-type model) and the base material (hyperelastic, Yeoh).

Check out this cool video of a fully dynamic, fully active, isometric contraction!

[![Dynamic isometric contraction of a muscle block with 30 degrees initial pennation angle.](reports/dynamic-isometric-contraction.png)](https://youtu.be/CCTiSV1Vl7o)




## How to use it?

- Make sure you have properly set up deal.II v9.3 (or newer) and its dependencies. For more information, visit [their website](http://www.dealii.org).
- Clone the repository using your SSH keys: ```git clone git@github.com:javieralmonacid/flexodeal.git```. Download the repository as a .zip if you do not want to track your changes or you do not have a GitHub account.
- Navigate to the Flexodeal folder using ```cd```.
- Call CMake:
    - In release mode: ```cmake . -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=<path/to/deal.II>```.
    - In debug mode: ```cmake . -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=<path/to/deal.II>```.
- Call make. A simple call to ```make``` should suffice.
- Run the code. This either achieved by calling ``` make run``` or ```./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation.dat ```. A folder with the current timestamp will be created. This is where the results of your execution will be stored.

## Latest line count

```
Tue May 14 18:19:05 PDT 2024

>> cloc --exclude-dir=2024* --exclude-lang=JSON,XML,make .
    
    11 text files.
    11 unique files.                              
     7 files ignored.

github.com/AlDanial/cloc v 1.90  T=0.02 s (267.5 files/s, 313641.8 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              1            680           1374           3142
TeX                              2            119             83            376
Markdown                         1             14              0             35
CMake                            1              6             15             18
-------------------------------------------------------------------------------
SUM:                             5            819           1472           3571
-------------------------------------------------------------------------------

```
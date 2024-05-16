# Flexodeal Lite
This is a reduced version of the [Flexodeal](https://github.com/javieralmonacid/flexodeal) library which has been structured to follow [deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html) guidelines. 

It is intended to simulate the deformation of a block of muscle tissue (similar to a biopsy). The stress response has been simplified to include contributions from muscle fibres (Hill-type model) and the base material (hyperelastic, Yeoh).

Check out this [cool video](https://youtu.be/CCTiSV1Vl7o) of a fully dynamic, fully active, isometric contraction!

## How to use it?

- Make sure you have properly set up deal.II v9.3 (or newer) and its dependencies. For more information, visit [their website](http://www.dealii.org).
- Clone the repository using your SSH keys: ```git clone git@github.com:sfu-nml/flexodeal.git```. Download the repository as a .zip if you do not want to track your changes or you do not have a GitHub account.
- Navigate to the Flexodeal folder using ```cd```.
- Call CMake:
    - In release mode: ```cmake . -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=<path/to/deal.II>```.
    - In debug mode: ```cmake . -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=<path/to/deal.II>```.
- Call make. A simple call to ```make``` should suffice.
- Run the code. This either achieved by calling ``` make run``` or ```./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation.dat ```. A folder with the current timestamp will be created. This is where the results of your execution will be stored.

## How long does it take to run?

In an Intel i5-9600K (3.70 GHz x 6) machine the code with its default settings (dynamic) takes about 6 minutes to complete 500 ms of simulation. In turn, the quasi-static code (`set Type of simulation = quasi-static`) takes only 18 seconds! That speaks volumes about the nonlinearity of the dynamic problem.

## Latest line count

```
Wed May 15 21:37:28 PDT 2024

>> cloc --exclude-dir=2024* --exclude-lang=JSON,XML,make .
    
    18 text files.
    18 unique files.                              
    14 files ignored.

github.com/AlDanial/cloc v 1.74  T=0.05 s (105.1 files/s, 117705.1 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              1            696           1081           3160
TeX                              1             84             56            255
Visual Basic                     1             35              0            148
Markdown                         1             14              0             33
CMake                            1              6             15             18
-------------------------------------------------------------------------------
SUM:                             5            835           1152           3614
-------------------------------------------------------------------------------

```

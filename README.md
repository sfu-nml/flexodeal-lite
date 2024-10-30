# Flexodeal Lite
This is a reduced version of the [Flexodeal](https://github.com/javieralmonacid/flexodeal) library which has been structured to follow [deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html) guidelines. 

It is intended to simulate the deformation of a block of muscle tissue (similar to a biopsy). The stress response has been simplified to include contributions from muscle fibres (Hill-type model) and the base material (hyperelastic, Yeoh).

Check out this [cool video](https://youtu.be/CCTiSV1Vl7o) of a fully dynamic, fully active, isometric contraction!

## How to use it?

- Make sure you have properly set up deal.II v9.3 (or newer) and its dependencies. For more information, visit [their website](http://www.dealii.org).
- Clone the repository: ```git clone https://github.com/sfu-nml/flexodeal-lite.git```. Download the repository as a .zip if you do not want to track your changes or you do not have a GitHub account.
- Navigate to the Flexodeal folder using ```cd```.
- Call CMake:
    - In release mode: ```cmake . -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=<path/to/deal.II>```.
    - In debug mode: ```cmake . -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=<path/to/deal.II>```.
- Call make. A simple call to ```make``` should suffice.
- Run the code. This either achieved by calling ``` make run``` or ```./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation.dat ```. A folder with the current timestamp will be created. This is where the results of your execution will be stored.

## How long does it take to run?

In an Intel i5-9600K (3.70 GHz x 6) machine the code with its default settings (dynamic) takes about 3 minutes to complete 500 ms of simulation. In turn, the quasi-static code (`set Type of simulation = quasi-static`) takes only 18 seconds! That speaks volumes about the nonlinearity of the dynamic problem.

## Latest line count (in main branch)

```
Tue 29 Oct 2024 06:04:56 PM PDT

>> cloc --exclude-dir=2024* --exclude-lang=JSON,XML,make .
    
    9 text files.
       9 unique files.                              
       7 files ignored.

github.com/AlDanial/cloc v 1.90  T=0.02 s (197.3 files/s, 338090.2 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              1            707           1078           3271
Markdown                         1             12              0             35
CMake                            1              6             15             18
-------------------------------------------------------------------------------
SUM:                             3            725           1093           3324
-------------------------------------------------------------------------------

```

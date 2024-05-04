# Flexodeal Lite
This is a reduced version of the [Flexodeal](https://github.com/javieralmonacid/flexodeal) library which has been structured to follow [deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html) guidelines. 

It is intended to simulate the deformation of a block of muscle tissue (similar to a biopsy). The stress response has been simplified to include contributions from muscle fibres (Hill-type model) and the base material (hyperelastic, Yeoh).

Check out this cool video of a fully dynamic, fully active, isometric contraction!

[![Dynamic isometric contraction of a muscle block with 30 degrees initial pennation angle.](reports/dynamic-isometric-contraction.png)](https://youtu.be/CCTiSV1Vl7o)




## How to use it?

- Make sure you have properly set up deal.II v9.3 and its dependencies. For more information, visit [their website](http://www.dealii.org).
- Clone the repository using your SSH keys: ```git clone git@github.com:javieralmonacid/flexodeal.git```. Download the repository as a .zip if you do not want to track your changes or you do not have a GitHub account.
- Navigate to the Flexodeal folder using ```cd```.
- Call CMake:
    - In release mode: ```cmake . -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=<path/to/deal.II>```.
    - In debug mode: ```cmake . -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=<path/to/deal.II>```.
- Call make. A simple call to ```make``` should suffice.
- Run the code. This either achieved by calling ``` make run``` or ```./dynamic-muscle```. A folder with the current timestamp will be created. This is where the results of your execution will be stored.


## Latest line count

```
Wed Apr 24 15:05:40 PDT 2024

>> cloc --exclude-dir=2024* --exclude-lang=JSON,XML,make .
    
    72 text files.
      63 unique files.                              
      47 files ignored.

github.com/AlDanial/cloc v 1.74  T=0.11 s (236.4 files/s, 72052.8 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              3            778           1431           3972
C                                2            101             59            472
CMake                           18             73             82            327
TeX                              1             84             56            255
Visual Basic                     1             35              0            148
Markdown                         1             15              0             35
-------------------------------------------------------------------------------
SUM:                            26           1086           1628           5209
-------------------------------------------------------------------------------

```
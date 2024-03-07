# Flexodeal Lite
This is a reduced version of the [Flexodeal](https://github.com/javieralmonacid/flexodeal) library which has been structured to follow [deal.II tutorial](https://www.dealii.org/current/doxygen/deal.II/Tutorial.html) guidelines. 

It is intended to simulate the deformation of a block of muscle tissue (similar to a biopsy). The stress response has been simplified to include contributions from muscle fibres (Hill-type model) and the base material (hyperelastic, Yeoh).

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
Wed Mar  6 16:37:22 PST 2024

>> cloc --exclude-dir=2024* --exclude-lang=JSON,XML,make .
    
    4392 text files.
    4345 unique files.                                          
    4362 files ignored.

github.com/AlDanial/cloc v 1.74  T=3.42 s (9.1 files/s, 2665.0 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
C++                              4            858           1409           4285
C                                3            215            117            971
CMake                           21            112             84            465
TeX                              1             84             56            255
Visual Basic                     1             35              0            148
Markdown                         1              6              0             33
-------------------------------------------------------------------------------
SUM:                            31           1310           1666           6157
-------------------------------------------------------------------------------

```
#!/bin/bash

./dynamic-muscle parameters_0.prm control_points_strain.dat control_points_activation.dat
./dynamic-muscle parameters_1_40.prm control_points_strain.dat control_points_activation.dat
./dynamic-muscle parameters_1_80.prm control_points_strain.dat control_points_activation.dat
./dynamic-muscle parameters_1_160.prm control_points_strain.dat control_points_activation.dat
./dynamic-muscle parameters_1_320.prm control_points_strain.dat control_points_activation.dat
./dynamic-muscle parameters_1_640.prm control_points_strain.dat control_points_activation.dat
./dynamic-muscle parameters_1_1280.prm control_points_strain.dat control_points_activation.dat
#!/bin/bash
sleep 5
./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation_20.dat
./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation_40.dat
./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation_60.dat
./dynamic-muscle parameters.prm control_points_strain.dat control_points_activation_80.dat

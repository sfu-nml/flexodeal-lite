enable_testing()

set(BINARY "${CMAKE_BINARY_DIR}/flexodeal-lite")
set(ROOT   "${CMAKE_SOURCE_DIR}")

# Find timeout command (gtimeout on macOS, timeout on Linux)
find_program(TIMEOUT_CMD NAMES gtimeout timeout)

macro(add_smoke_test NAME COMMAND_STR)
  add_test(
    NAME smoke/${NAME}
    COMMAND bash -c "
      mkdir -p ${CMAKE_BINARY_DIR}/smoke_out/${NAME} && ${TIMEOUT_CMD} 10 ${COMMAND_STR} -OUTPUT_DIR=${CMAKE_BINARY_DIR}/smoke_out/${NAME}
      CODE=$?
      [ $CODE -eq 0 ] || [ $CODE -eq 124 ] && exit 0 || exit $CODE
    "
  )
  set_tests_properties(smoke/${NAME} PROPERTIES
    TIMEOUT  20
    LABELS   "smoke"
    WORKING_DIRECTORY  "${ROOT}"
  )
endmacro()

add_smoke_test(DEFAULT
  "${BINARY}"
)

add_smoke_test(IC
  "${BINARY} \
    -PARAMETERS=${ROOT}/examples/IC_parameters.prm \
    -BDY_STRAIN=${ROOT}/examples/IC_strain.dat \
    -ACTIVATION=${ROOT}/examples/IC_activation.dat"
)

add_smoke_test(ICWP
  "${BINARY} \
    -PARAMETERS=${ROOT}/examples/ICWP_parameters.prm \
    -BDY_STRAIN=${ROOT}/examples/ICWP_strain.dat \
    -ACTIVATION=${ROOT}/examples/ICWP_activation.dat \
    -PRESSURE=${ROOT}/examples/ICWP_pressure.dat"
)

add_smoke_test(AL
  "${BINARY} \
    -PARAMETERS=examples/AL_parameters.prm \
    -BDY_STRAIN=examples/AL_strain.dat \
    -ACTIVATION=examples/AL_activation.dat"
)

add_smoke_test(CC
  "${BINARY} \
    -PARAMETERS=examples/CC_parameters.prm \
    -BDY_STRAIN=examples/CC_strain.dat \
    -ACTIVATION=examples/CC_activation.dat"
)

add_smoke_test(GE
  "${BINARY} \
    -PARAMETERS=${ROOT}/examples/GE_parameters.prm \
    -BDY_STRAIN=${ROOT}/examples/GE_strain.dat \
    -ACTIVATION=${ROOT}/examples/GE_activation.dat \
    -MARKERS=${ROOT}/examples/GE_markers.dat"
)
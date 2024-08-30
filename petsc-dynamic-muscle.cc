/* ---------------------------------------------------------------------
 *
 * Flexodeal Lite
 * Copyright (C) 2024 Neuromuscular Mechanics Laboratory
 *
 * This file is part of the Flexodeal library
 *
 * The Flexodeal library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of Flexodeal.
 *
 * ---------------------------------------------------------------------
 * 
 * Author: Javier Almonacid
 *         PhD Candidate, Applied and Computational Mathematics
 *         Neuromuscular Mechanics Laboratory (NML)
 *         Simon Fraser University
 *         Spring 2024
 * 
 * This software has been created based on the "muscle code" developed by
 * members of the NML since 2014:
 * 
 *         Ryan N. Konno
 *         Cassidy Tam
 *         Sebastian A. Dominguez-Rivera
 *         Stephanie A. Ross
 *         David Ryan
 *         Hadi Rahemi
 *         Prof. James M. Wakeling (SFU Biomedical Physiology and Kinesiology)
 *         Prof. Nilima Nigam (SFU Mathematics)
 * 
 * Furthermore, the structure of the code is based on deal.II's step-44 tutorial:
 * 
 *         Pelteret, J.-P., & McBride, A. (2012). The deal.II tutorial step-44: 
 *         Three-field formulation for non-linear solid mechanics. 
 *         Zenodo. https://doi.org/10.5281/zenodo.439772
 * 
 * Some comments have been preserved from step-44 to increase readability of the
 * code. The mathematical details of the muscle model in use are available here:
 * 
 *         Almonacid, J. A., Domínguez-Rivera, S. A., Konno, R. N., Nigam, N., 
 *         Ross, S. A., Tam, C., & Wakeling, J. M. (2024). 
 *         A three-dimensional model of skeletal muscle tissues. 
 *         SIAM Journal on Applied Mathematics, S538-S566.
 * 
 * 
 */

// First, we include all the headers necessary for this code. All the headers
// (with the exception of fe_dgq.h) have already been discussed in step-44 and step-40.
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h> 
#include <deal.II/lac/sparsity_tools.h> 
#include <deal.II/lac/full_matrix.h>

#include <deal.II/lac/petsc_block_sparse_matrix.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_snes.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>


// Then, we place all functions and classes inside a namespace of its own.
namespace Flexodeal
{
  using namespace dealii;

  // @sect3{Run-time parameters}
  //
  // There are several parameters that can be set in the code so we set up a
  // ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
    // @sect4{Finite Element system}

    // As mentioned in the introduction, a different order interpolation should
    // be used for the displacement $\mathbf{u}$ than for the pressure
    // $\widetilde{p}$ and the dilatation $\widetilde{J}$.  Choosing
    // $\widetilde{p}$ and $\widetilde{J}$ as discontinuous (constant) functions
    // at the element level leads to the mean-dilatation method. The
    // discontinuous approximation allows $\widetilde{p}$ and $\widetilde{J}$ to
    // be condensed out and a classical displacement based method is recovered.
    // Here we specify the polynomial order used to approximate the solution.
    // The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;
      std::string  type_of_simulation;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };


    void FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order",
                          "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");

        prm.declare_entry("Type of simulation",
                          "quasi-static",
                          Patterns::Selection("quasi-static|dynamic"),
                          "Type of simulation (quasi-static or dynamic)");
      }
      prm.leave_subsection();
    }

    void FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order  = prm.get_integer("Quadrature order");
        type_of_simulation = prm.get("Type of simulation");
      }
      prm.leave_subsection();
    }

    // @sect4{Geometry}

    // Next, we process the parameters related to the geometry of the problem.
    // To keep things simple, we consider a block of muscle tissue of given
    // length, width, and height. Note that this structure will probably have
    // to be adjusted whenever a different geometry is considered.
    struct Geometry
    {
      unsigned int global_refinement;
      double       length;
      double       width;
      double       height;
      double       scale;
      double       p_p0;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Global refinement",
                          "2",
                          Patterns::Integer(0),
                          "Global refinement level");
        
        prm.declare_entry("Length", "1.0",
                          Patterns::Double(0.0),
                          "Length in the x direction");

        prm.declare_entry("Width", "1.0",
                          Patterns::Double(0.0),
                          "Width in the y direction");

        prm.declare_entry("Height", "1.0",
                          Patterns::Double(0.0),
                          "Height in the z direction");

        prm.declare_entry("Grid scale",
                          "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        prm.declare_entry("Pressure ratio p/p0",
                          "100",
                          Patterns::Selection("0|20|40|60|80|100"),
                          "Ratio of applied pressure to reference pressure");
      }
      prm.leave_subsection();
    }

    void Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        global_refinement = prm.get_integer("Global refinement");
        length            = prm.get_double("Length"); 
        width             = prm.get_double("Width"); 
        height            = prm.get_double("Height");
        scale             = prm.get_double("Grid scale");
        p_p0              = prm.get_double("Pressure ratio p/p0");
      }
      prm.leave_subsection();
    }

    // @sect4{Muscle properties}

    // Then, we process all the intrinsic properties of muscle tissue.
    // In this model, muscle is viewed as a bundle of muscle fibres 
    // surrounded by a base material.
    struct MuscleProperties
    {
      double muscle_density;

      // Fibre properties
      double max_iso_stress_muscle; 
      double kappa_muscle;
      double max_strain_rate; 
      double muscle_fibre_orientation_x; 
      double muscle_fibre_orientation_y; 
      double muscle_fibre_orientation_z; 

      // Base material properties
      double max_iso_stress_basematerial;
      double muscle_basematerial_factor; 
      double muscle_basemat_c1;
      double muscle_basemat_c2;
      double muscle_basemat_c3;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void MuscleProperties::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Materials");
      {
        prm.declare_entry("Muscle density", "1060",
                          Patterns::Double(),
                          "Muscle tissue density");

        // Fibre properties
        prm.declare_entry("Sigma naught muscle", "2.0e5",
                          Patterns::Double(),
                          "Muscle maximum isometric stress");

        prm.declare_entry("Bulk modulus muscle", "1.0e6",
                          Patterns::Double(),
                          "Muscle kappa value");

        prm.declare_entry("Max strain rate", "0.0",
                          Patterns::Double(),
                          "Maximum muscle fibre strain rate");
        
        prm.declare_entry("Muscle x component", "1.0",
                          Patterns::Double(0.0),
                          "Muscle fibre orientation x direction");

        prm.declare_entry("Muscle y component", "0.0",
                          Patterns::Double(0.0),
                          "Muscle fibre orientation y direction");

        prm.declare_entry("Muscle z component", "0.0",
                          Patterns::Double(0.0),
                          "Muscle fibre orientation z direction");
        
        // Base material properties
        prm.declare_entry("Sigma naught base material", "2.0e5",
                          Patterns::Double(),
                          "Base material maximum isometric stress");
        
        prm.declare_entry("Muscle base material factor", "1.0e1",
                          Patterns::Double(),
                          "Fictitious muscle base material multiplier");
        
        prm.declare_entry("Muscle base material constant 1", "0.0",
                          Patterns::Double(),
                          "Muscle base material constant 1");

        prm.declare_entry("Muscle base material constant 2", "0.0",
                          Patterns::Double(),
                          "Muscle base material constant 2");

        prm.declare_entry("Muscle base material constant 3", "0.0",
                          Patterns::Double(),
                          "Muscle base material constant 3");
      }
      prm.leave_subsection();
    }

    void MuscleProperties::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Materials");
      {
        muscle_density          = prm.get_double("Muscle density");

        max_iso_stress_muscle   = prm.get_double("Sigma naught muscle");
        kappa_muscle            = prm.get_double("Bulk modulus muscle");
        max_strain_rate         = prm.get_double("Max strain rate");
        muscle_fibre_orientation_x  = prm.get_double("Muscle x component"); 
        muscle_fibre_orientation_y  = prm.get_double("Muscle y component"); 
        muscle_fibre_orientation_z  = prm.get_double("Muscle z component");

        max_iso_stress_basematerial = prm.get_double("Sigma naught base material");
        muscle_basematerial_factor  = prm.get_double("Muscle base material factor");
        muscle_basemat_c1 = prm.get_double("Muscle base material constant 1"); 
        muscle_basemat_c2 = prm.get_double("Muscle base material constant 2"); 
        muscle_basemat_c3 = prm.get_double("Muscle base material constant 3"); 
      }
      prm.leave_subsection();
    }

    // @sect4{Linear solver}

    // Next, we choose both solver and preconditioner settings.  The use of an
    // effective preconditioner is critical to ensure convergence when a large
    // nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;
      bool        use_static_condensation;
      std::string preconditioner_type;
      double      preconditioner_relaxation;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type",
                          "CG",
                          Patterns::Selection("GMRES|CG|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual",
                          "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry(
          "Max iteration multiplier",
          "1",
          Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Use static condensation",
                          "true",
                          Patterns::Bool(),
                          "Solve the full block system or a reduced problem");

        prm.declare_entry("Preconditioner type",
                          "ssor",
                          Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation",
                          "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
      }
      prm.leave_subsection();
    }

    void LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin                  = prm.get("Solver type");
        tol_lin                   = prm.get_double("Residual");
        max_iterations_lin        = prm.get_double("Max iteration multiplier");
        use_static_condensation   = prm.get_bool("Use static condensation");
        preconditioner_type       = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
      }
      prm.leave_subsection();
    }

    // @sect4{Nonlinear solver}

    // A Newton scheme is used to solve the nonlinear system of governing
    // equations. Below we process the stopping criteria for the solver.
    struct NonlinearSolver
    {
      std::string  type_nonlinear_solver;
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Nonlinear solver type", "classicNewton",
                          Patterns::Selection("classicNewton|acceleratedNewton"),
                          "Type of nonlinear iteration");

        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        type_nonlinear_solver = prm.get("Nonlinear solver type");
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f             = prm.get_double("Tolerance force");
        tol_u             = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }

    // @sect4{Time}

    // Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t;
      double end_time;

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);
    };

    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time step size",
                          "0.1",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t  = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

    // @sect4{PrescribedDisplacement}

    // Set the parameters for the prescribed displacement. Because the profile 
    // itself is given from control_points_strain.dat file, the only thing we 
    // keep track here is the ID of the face we are pulling/pushing from. This
    // is critical to correctly identify the face for which forces will be
    // reported as a time series.
    struct PrescribedDisplacement
    {
      unsigned int pulling_face_id;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void PrescribedDisplacement::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Prescribed displacement");
      {
        prm.declare_entry("Pulling face ID", "1",
                          Patterns::Integer(0),
                          "Boundary ID of face being pulled/pushed");
      }
      prm.leave_subsection();
    }

    void PrescribedDisplacement::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Prescribed displacement");
      {
        pulling_face_id  = prm.get_integer("Pulling face ID");
      }
      prm.leave_subsection();
    }

    // @sect4{Measuring locations}
    
    // We select three points in the geometry at which we will
    // output traces of displacement.
    struct MeasuringLocations
    {
      double x_left;
      double y_left;
      double z_left;
      double x_mid;
      double y_mid;
      double z_mid;
      double x_right;
      double y_right;
      double z_right;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Parameters::MeasuringLocations::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Measuring locations");
      {
        prm.declare_entry("Left X", "0.0",
                          Patterns::Double(),
                          "Left measuring point, X coordinate");

        prm.declare_entry("Left Y", "0.0",
                          Patterns::Double(),
                          "Left measuring point, Y coordinate");

        prm.declare_entry("Left Z", "0.0",
                          Patterns::Double(),
                          "Left measuring point, Z coordinate");

        prm.declare_entry("Mid X", "0.0",
                          Patterns::Double(),
                          "Mid measuring point, X coordinate");

        prm.declare_entry("Mid Y", "0.0",
                          Patterns::Double(),
                          "Mid measuring point, Y coordinate");

        prm.declare_entry("Mid Z", "0.0",
                          Patterns::Double(),
                          "Mid measuring point, Z coordinate");

        prm.declare_entry("Right X", "0.0",
                          Patterns::Double(),
                          "Right measuring point, X coordinate");

        prm.declare_entry("Right Y", "0.0",
                          Patterns::Double(),
                          "Right measuring point, Y coordinate");

        prm.declare_entry("Right Z", "0.0",
                          Patterns::Double(),
                          "Right measuring point, Z coordinate");
      }
      prm.leave_subsection();
    }

    void Parameters::MeasuringLocations::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Measuring locations");
      {
        x_left = prm.get_double("Left X");
        y_left = prm.get_double("Left Y");
        z_left = prm.get_double("Left Z");
        x_mid = prm.get_double("Mid X");
        y_mid = prm.get_double("Mid Y");
        z_mid = prm.get_double("Mid Z");
        x_right = prm.get_double("Right X");
        y_right = prm.get_double("Right Y");
        z_right = prm.get_double("Right Z");
      }
      prm.leave_subsection();
    }

    // @sect4{All parameters}

    // Finally we consolidate all of the above structures into a single
    // container that holds all of our run-time selections.
    struct AllParameters : public FESystem,
                           public Geometry,
                           public MuscleProperties,
                           public LinearSolver,
                           public NonlinearSolver,
                           public Time,
                           public PrescribedDisplacement,
                           public MeasuringLocations

    {
      AllParameters(const std::string &input_file);

      static void declare_parameters(ParameterHandler &prm);

      void parse_parameters(ParameterHandler &prm);

      // We decide to keep track of the parameter handler
      // object since we want to call prm.print_parameters()
      // after the constructor has been called and we need
      // to first set up the directories with the current
      // timestamp.
      ParameterHandler prm;
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      MuscleProperties::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
      PrescribedDisplacement::declare_parameters(prm);
      MeasuringLocations::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      MuscleProperties::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
      PrescribedDisplacement::parse_parameters(prm);
      MeasuringLocations::parse_parameters(prm);
    }
  } // namespace Parameters

  // @sect3{Time class}

  // A simple class to store time data. Its functioning is transparent so no
  // discussion is necessary. For simplicity we assume a constant time step
  // size.
  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0)
      , time_previous(0.0)
      , time_current(0.0)
      , time_end(time_end)
      , delta_t(delta_t)
    {}

    virtual ~Time() = default;

    double current() const
    {
      return time_current;
    }
    double previous() const
    {
      return time_previous;
    }
    double end() const
    {
      return time_end;
    }
    double get_delta_t() const
    {
      return delta_t;
    }
    unsigned int get_timestep() const
    {
      return timestep;
    }
    void increment()
    {
      time_previous = time_current;
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_previous;
    double       time_current;
    const double time_end;
    const double delta_t;
  };

  // @sect3{Function from tabular data and interpolation tools}

  // This class can be used to read data from a table of
  // x,y values (separated by a space or tab) and construct
  // a continuous function from them using linear interpolation.
  // This will be particularly useful to read activations and 
  // muscle length profiles from data. Although it may seem like
  // an overkill, this class can also be used to implement simple
  // linear ramps. For instance, a linear ramp between (0,0) and 
  // (0.5, 1.0) only requires a .dat file with 2 lines:
  //
  //                  0.0 0.0
  //                  0.5 1.0
  //
  class TabularFunction
  {
  public:
    TabularFunction(const std::string filename)
    {
      initialize_map(filename);
    }

    double operator()(const double t);

  private:
    std::map<double,double> table_values;
    void initialize_map(const std::string filename);
  };

  // Read data (control points) and store them in the table_values map
  void TabularFunction::initialize_map(const std::string filename)
  {
    std::ifstream infile(filename);

    // Raise an exception if file cannot be open (perhaps it does not
    // even exist!).
    if (infile.fail())
      throw std::invalid_argument("Cannot open file: " + filename +  
            ". Make sure the file exists and it has read permissions.");
    
    double x, y;
    while (infile >> x >> y)
      table_values.insert({x, y});
  }

  // Evaluate the data:
  // - Interpolate if not found and t (independent variable) 
  //   is in data range
  // - Retrieve the original y coordinate if found
  // - Output a constant value if t exceeds data range
  double TabularFunction::operator()(const double t)
  {
    double out = 1000000; /*Bogus value*/
    auto iter_t = table_values.find(t);
    
    if (iter_t == table_values.end()) /* Value not found: interpolate! */
    {
      if (t <= table_values.rbegin()->first)
      {
        auto t1 = table_values.upper_bound(t);
        auto t0 = std::prev(t1);
        out = ((t1->second - t0->second)/(t1->first - t0->first)) * (t - t0->first) + t0->second;
      }
      else
        out = table_values.rbegin()->second; /* Constant after last point */
    }
    else /* Value found! Return this value */
      out = iter_t->second;

    return out;
  }

  // @sect3{Muscle tissue within a three-field formulation}

  // Muscle can be described as a quasi-incompressible fibre-reinforced
  // material. Similar to the <code>Material_Compressible_Neo_Hook_Three_Field</code>
  // in step-44, the material here can be described by a strain-energy 
  // function $ \Psi = \Psi_{\text{iso}}(\overline{\mathbf{b}}) + 
  // \Psi_{\text{vol}}(\widetilde{J})$, where $\Psi_{iso}$ is composed of a 
  // fibre and a base material component $\Psi_{fibre}$ and $\Psi_{base}$, 
  // respectively.
  // 
  // In the current configuration, the fibre component takes the form
  // $\boldsymbol{\sigma} = \sigma_{Hill} \mathbf{a} \otimes \mathbf{a}$,
  // where $\sigma_{Hill} = \sigma_0 \left\{ a(t) \sigma_L(\lambda) 
  // \sigma_V(\epsilon) + \sigma_P(\lambda) \right\}$.
  //
  // In turn, the base material component is simply given by a Yeoh SEF.
  //
  // A note on quasi-incompressibility of muscle: 
  // 
  //  Technically, muscle is compressible (Baskin & Paolini 1966). However, the 
  //  magnitude of this volume change is so small that most physiologists would
  //  consider muscle as "incompressible". Therefore, the volume changes that we
  //  expect here will (and have to) be small and the dilation J will hover around
  //  the value of 1. From a numerical perspective, though these changes are small,
  //  they are extremely important to prevent locking effects.
  template <int dim>
  class Muscle_Tissues_Three_Field
  {
  public:
    Muscle_Tissues_Three_Field(const std::string type_contraction,
                               const double max_iso_stress_muscle,
                               const double kappa_muscle,
                               const double max_strain_rate,
                               const double initial_fibre_orientation_x,
                               const double initial_fibre_orientation_y,
                               const double initial_fibre_orientation_z,
                               const double max_iso_stress_basematerial,
                               const double muscle_basematerial_factor,
                               const double muscle_basemat_c1,
                               const double muscle_basemat_c2,
                               const double muscle_basemat_c3)
      :
      type_of_contraction(type_contraction),
      sigma_naught_muscle(max_iso_stress_muscle),
      kappa_muscle(kappa_muscle),
      strain_rate_naught(max_strain_rate),
      initial_fibre_orientation({initial_fibre_orientation_x, 
                                 initial_fibre_orientation_y, 
                                 initial_fibre_orientation_z}),
      sigma_naught_basematerial(max_iso_stress_basematerial),
      s_base_muscle(muscle_basematerial_factor),
      c1_basematerial_muscle(muscle_basemat_c1),
      c2_basematerial_muscle(muscle_basemat_c2),
      c3_basematerial_muscle(muscle_basemat_c3),
      /* Physiological variables */
      stretch_bar(1.0),
      strain_rate_bar(0.0),
      fibre_time_activation(0.0),
      orientation(initial_fibre_orientation),
      /* Mechanical variables */
      det_F(1.0),
      p_tilde(0.0),
      J_tilde(1.0),
      b_bar(Physics::Elasticity::StandardTensors<dim>::I),
      trace_b_bar(3.0),
      trace_d(0.0),
      delta_t(0.0)
      {
        Assert(kappa_muscle > 0, 
               ExcMessage("Bulk modulus must be positive!"));
        Assert(initial_fibre_orientation.norm() != 0, 
               ExcMessage("Initial fibre orientation must be a nonzero vector!"))
      }

    // We update the material model with various deformation dependent data
    // based on $F$ and the pressure $\widetilde{p}$ and dilatation
    // $\widetilde{J}$, and at the end of the function include a physical
    // check for internal consistency:
    void update_material_data(const Tensor<2, dim> &F,
                              const double          p_tilde_in,
                              const double          J_tilde_in,
                              const double          fibre_time_activation_in,
                              const Tensor<2, dim> &grad_velocity,
                              const double          delta_t_in)
    {
      // First compute the determinant of the deformation tensor
      // and stop the program immediately if it is negative.
      det_F                      = determinant(F);
      AssertThrow(det_F > 0, ExcInternalError());
      
      // Then, update the rest of the variables.
      p_tilde                    = p_tilde_in;
      J_tilde                    = J_tilde_in;
      fibre_time_activation      = fibre_time_activation_in;
      delta_t                    = delta_t_in;

      const Tensor<2, dim> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
      b_bar                      = Physics::Elasticity::Kinematics::b(F_bar);
      trace_b_bar                = first_invariant(b_bar);
      
      // Update stretch_bar and strain_rate_bar using the current Newton iterate.
      const SymmetricTensor<2,dim> symm_grad_velocity = Physics::Elasticity::Kinematics::d(F,grad_velocity);
      const Tensor<2,dim> dev_symm_grad_velocity      = Physics::Elasticity::StandardTensors<dim>::dev_P * symm_grad_velocity;
      trace_d         = first_invariant(symm_grad_velocity);
      orientation     = F_bar * initial_fibre_orientation;
      stretch_bar     = std::sqrt(orientation * orientation);
      strain_rate_bar = (1.0 / strain_rate_naught) * 
                         orientation * (dev_symm_grad_velocity * orientation) / stretch_bar;
    }

    // The second function determines the Kirchhoff stress $\boldsymbol{\tau}
    // = \boldsymbol{\tau}_{\textrm{iso}} + \boldsymbol{\tau}_{\textrm{vol}}$
    SymmetricTensor<2, dim> get_tau()
    {
      return get_tau_iso() + get_tau_vol();
    }

    // The following set of functions determine each contribution to the
    // Kirchhoff stress in detail. We make this functions public as they
    // need to be accessed by PointHistory when computing forces and 
    // energies. The first one determines the volumetric Kirchhoff stress 
    // $\boldsymbol{\tau}_{\textrm{vol}}$:
    SymmetricTensor<2, dim> get_tau_vol() const
    {
      return p_tilde * det_F * Physics::Elasticity::StandardTensors<dim>::I;
    }

    // Next, determine the isochoric Kirchhoff stress
    // $\boldsymbol{\tau}_{\textrm{iso}} =
    // \mathcal{P}:\overline{\boldsymbol{\tau}}$:
    SymmetricTensor<2, dim> get_tau_iso() const
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_bar();
    }

    // Just like the SEF of the system, the isochoric
    // Kirchhoff stress is made of two parts:
    // a fibre component and a base material component.
    // In particular, we decide to subdivide the
    // fibre component into its active and passive
    // components.
    SymmetricTensor<2,dim> get_tau_iso_muscle_active()
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_muscle_active_bar();
    }

    SymmetricTensor<2,dim> get_tau_iso_muscle_passive()
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_muscle_passive_bar();
    }

    SymmetricTensor<2,dim> get_tau_iso_muscle_basematerial()
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P * get_tau_muscle_basematerial_bar();
    }

    // The fourth-order elasticity tensor in the spatial setting
    // $\mathfrak{c}$ is calculated from the SEF $\Psi$ as $ J
    // \mathfrak{c}_{ijkl} = F_{iA} F_{jB} \mathfrak{C}_{ABCD} F_{kC} F_{lD}$
    // where $ \mathfrak{C} = 4 \frac{\partial^2 \Psi(\mathbf{C})}{\partial
    // \mathbf{C} \partial \mathbf{C}}$
    SymmetricTensor<4, dim> get_Jc() const
    {
      return get_Jc_vol() + get_Jc_iso();
    }

    // Derivative of the volumetric free energy with respect to
    // $\widetilde{J}$ return $\frac{\partial
    // \Psi_{\text{vol}}(\widetilde{J})}{\partial \widetilde{J}}$
    double get_dPsi_vol_dJ() const
    {
      return (kappa_muscle / 2.0) * (J_tilde - 1.0 / J_tilde);
    }

    // Second derivative of the volumetric free energy wrt $\widetilde{J}$. We
    // need the following computation explicitly in the tangent so we make it
    // public.  We calculate $\frac{\partial^2
    // \Psi_{\textrm{vol}}(\widetilde{J})}{\partial \widetilde{J} \partial
    // \widetilde{J}}$
    double get_d2Psi_vol_dJ2() const
    {
      return ((kappa_muscle / 2.0) * (1.0 + 1.0 / (J_tilde * J_tilde)));
    }

    // The next few functions return various data that we choose to store with
    // the material:
    double get_det_F() const
    {
      return det_F;
    }

    double get_p_tilde() const
    {
      return p_tilde;
    }

    double get_J_tilde() const
    {
      return J_tilde;
    }

    double get_stretch() const
    {
      return std::pow(det_F, 1/3) * stretch_bar;
    }

    double get_stretch_bar() const
    {
      return stretch_bar;
    }

    double get_strain_rate() const
    {
      return std::pow(det_F, 1/3) * (strain_rate_bar + (1.0/dim) * (trace_d/strain_rate_naught) * stretch_bar);
    }

    double get_strain_rate_bar() const
    {
      return strain_rate_bar;
    }

    Tensor<1, dim> get_orientation() const
    {
      return std::pow(det_F, 1/3) * orientation;
    }

  protected:
    // Define constitutive model parameters
    const std::string       type_of_contraction;
    const double            sigma_naught_muscle;
    const double            kappa_muscle;
    const double            strain_rate_naught;
    const Tensor<1, dim>    initial_fibre_orientation;
    const double            sigma_naught_basematerial;
    const double            s_base_muscle;
    const double            c1_basematerial_muscle;
    const double            c2_basematerial_muscle;
    const double            c3_basematerial_muscle;

    // Define physiological variables needed to evaluate the 
    // constitutive models
    double                  stretch_bar;
    double                  strain_rate_bar;
    double                  fibre_time_activation;
    Tensor<1, dim>          orientation;

    // Define mechanical variables
    double                  det_F;
    double                  p_tilde;
    double                  J_tilde;
    SymmetricTensor<2, dim> b_bar;
    double                  trace_b_bar;
    double                  trace_d;
    double                  delta_t;

    // The following functions are used internally in determining the result
    // of some of the public functions above.
    // First, determine the fictitious Kirchhoff stress
    // $\overline{\boldsymbol{\tau}}$:
    SymmetricTensor<2, dim> get_tau_bar() const
    {
      return get_tau_muscle_active_bar() + get_tau_muscle_passive_bar() + get_tau_muscle_basematerial_bar();
    }

    // Determine the contributions from active and passive muscle fibres,
    // as well as from the base material:
    SymmetricTensor<2, dim> get_tau_muscle_active_bar() const
    {
      const double active_level = fibre_time_activation;
      double sigma_active_muscle_fibre = 0.0;
      
      if (type_of_contraction == "quasi-static")
        sigma_active_muscle_fibre = sigma_naught_muscle * active_level * get_length_stress() * 1.0;
      else if (type_of_contraction == "dynamic")
        sigma_active_muscle_fibre = sigma_naught_muscle * active_level * get_length_stress() * get_strain_rate_stress();
      
      return (1.0 / std::pow(stretch_bar, 2)) * sigma_active_muscle_fibre *
               symmetrize(outer_product(orientation,orientation));
    }

    SymmetricTensor<2, dim> get_tau_muscle_passive_bar() const
    {
      const double sigma_passive_muscle_fibre = sigma_naught_muscle * get_passive_stress();
      return (1.0 / std::pow(stretch_bar,2)) * sigma_passive_muscle_fibre * 
                symmetrize(outer_product(orientation,orientation));
    }

    SymmetricTensor<2, dim> get_tau_muscle_basematerial_bar() const
    {
      return 2 * s_base_muscle * sigma_naught_basematerial * 
            (3 * c3_basematerial_muscle * std::pow(trace_b_bar - 3,2)
           + 2 * c2_basematerial_muscle * (trace_b_bar - 3) + c1_basematerial_muscle) * b_bar;
    }

    // Calculate the volumetric part of the tangent $J
    // \mathfrak{c}_\textrm{vol}$:
    SymmetricTensor<4, dim> get_Jc_vol() const
    {
      return p_tilde * det_F *
             (Physics::Elasticity::StandardTensors<dim>::IxI -
              (2.0 * Physics::Elasticity::StandardTensors<dim>::S));
    }

    // Calculate the isochoric part of the tangent $J
    // \mathfrak{c}_\textrm{iso}$:
    SymmetricTensor<4, dim> get_Jc_iso() const
    {
      const SymmetricTensor<2, dim> tau_bar = get_tau_bar();
      const SymmetricTensor<2, dim> tau_iso = get_tau_iso();
      const SymmetricTensor<4, dim> tau_iso_x_I =
        outer_product(tau_iso, Physics::Elasticity::StandardTensors<dim>::I);
      const SymmetricTensor<4, dim> I_x_tau_iso =
        outer_product(Physics::Elasticity::StandardTensors<dim>::I, tau_iso);
      const SymmetricTensor<4, dim> c_bar = get_c_bar();

      return (2.0 / dim) * trace(tau_bar) *
               Physics::Elasticity::StandardTensors<dim>::dev_P -
             (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso) +
             Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar *
               Physics::Elasticity::StandardTensors<dim>::dev_P;
    }

    // Calculate the fictitious elasticity tensor $\overline{\mathfrak{c}}$.
    // Note that because the material is no longer Neo-Hookean as in step-44,
    // these tensors are no longer zero. Moreover, care must be taken when
    // linearizing the elasticity equations as the velocity variable, which
    // is only present in dynamic simulations, introduces additional terms
    // to the tangent matrix.
    SymmetricTensor<4, dim> get_c_bar() const
    {
      return get_c_muscle_active_bar() + get_c_muscle_passive_bar() + get_c_muscle_basematerial_bar();
    }

    SymmetricTensor<4, dim> get_c_muscle_active_bar() const
    {
      const double activation_level = fibre_time_activation;
      const SymmetricTensor<2, dim> 
        orientation_x_orientation = symmetrize(outer_product(orientation,orientation));

      double first_term = 0.0, second_term = 0.0, third_term = 0.0;

      if (type_of_contraction == "quasi-static")
      {
        first_term  = - (2 / std::pow( stretch_bar ,4))
                    * sigma_naught_muscle
                    * activation_level
                    * get_length_stress()
                    * 1.0;

        second_term = (1 / std::pow( stretch_bar ,3))
                    * sigma_naught_muscle
                    * activation_level
                    * get_dlength_stress_dstretch() * 1.0;
        
        third_term = 0.0;
      }
      else if (type_of_contraction == "dynamic")
      {
        first_term  = - (2 / std::pow( stretch_bar ,4))
                      * sigma_naught_muscle
                      * activation_level
                      * get_length_stress()
                      * get_strain_rate_stress();
        
        second_term =   (1 / std::pow( stretch_bar ,3))
                      * sigma_naught_muscle
                      * activation_level
                      * get_dlength_stress_dstretch() * get_strain_rate_stress();
        
        third_term  =   (1 / std::pow( stretch_bar ,3))
                      * sigma_naught_muscle
                      * activation_level
                      * get_length_stress() * get_dstrain_rate_stress_dstrain_rate() * (1.0 / delta_t);
      }

      return (first_term + second_term + third_term) * 
        outer_product(orientation_x_orientation,orientation_x_orientation);
    }

    SymmetricTensor<4, dim> get_c_muscle_passive_bar() const
    {
      const SymmetricTensor<2, dim> 
        orientation_x_orientation = symmetrize(outer_product(orientation,orientation));
      
      const double first_term  = - (2 / std::pow( stretch_bar ,4))
                               * sigma_naught_muscle
                               * get_passive_stress();
      const double second_term =   (1 / std::pow( stretch_bar ,3))
                                * sigma_naught_muscle
                                * get_dpassive_stress_dstretch();

      return (first_term + second_term) * 
        outer_product(orientation_x_orientation,orientation_x_orientation);
    }

    SymmetricTensor<4, dim> get_c_muscle_basematerial_bar() const
    {
      return 4 * sigma_naught_basematerial * s_base_muscle * 
            (6 * c3_basematerial_muscle * (trace_b_bar - 3)
           + 2 * c2_basematerial_muscle) * outer_product(b_bar,b_bar);
    }

    // Finally, we define the stress relationships of muscle,
    // along with their derivatives. These are the "stress"
    // versions of the traditional force-length and 
    // force-velocity relationships used in the traditional 
    // Hill model of muscle.
    double get_length_stress() const
    {
      if (stretch_bar >= 0.4 && stretch_bar <= 1.75)
        return  (0.642587074375392 * sin(1.290128342448810 * stretch_bar + 0.629168420414746)
                +0.325979591577056 * sin(5.308969899884336 * stretch_bar + -4.520101562237307)
                +0.328204247867325 * sin(6.744187042136006 * stretch_bar + 1.689155892259429)
                +0.015388902741327 * sin(19.823676877725276 * stretch_bar + -7.386155292116579)
                +0.139240359517525 * sin(8.038287396059996 * stretch_bar + 2.543022326676525)
                +0.001801867529599 * sin(32.237736486095052 * stretch_bar + -6.454098315528945)
                +0.012560837549867 * sin(23.117614057963024 * stretch_bar + -2.643346778503341));
      else
        return 0.0;
    }

    double get_strain_rate_stress() const
    {
      if (strain_rate_bar < -1.2)
        return 0.0;
      else if (strain_rate_bar >= -1.2 && strain_rate_bar < -0.25)
        return 0.25792669408341773*std::pow(strain_rate_bar+1.2,3)
            + 0.14317485143460784*std::pow(strain_rate_bar+1.2,2)
            + 0.0*(strain_rate_bar+1.2)
            + 0.0;
      else if (strain_rate_bar >= -0.25 && strain_rate_bar < 0.0)
        return 29.825565394304522*std::pow(strain_rate_bar+0.25,3)
            + -0.9435495605479662*std::pow(strain_rate_bar+0.25,2)
            + 0.9703687419567255*(strain_rate_bar+0.25)
            + 0.3503552027590582;
      else if (strain_rate_bar >= 0.0 && strain_rate_bar < 0.05)
        return -3165.6847983144276*std::pow(strain_rate_bar-0.0,3)
            + 186.19612494819665*std::pow(strain_rate_bar-0.0,2)
            + 6.090887473114851*(strain_rate_bar-0.0)
            + 1.0;
      else if (strain_rate_bar >= 0.05 && strain_rate_bar < 0.75)
        return 0.6882206253246714*std::pow(strain_rate_bar-0.05,3)
            + -1.413963071288272*std::pow(strain_rate_bar-0.05,2)
            + 0.9678639805763023*(strain_rate_bar-0.05)
            + 1.3743240862369306;
      else// if (strain_rate_bar >= 0.75)
          return 1.5950466421954534;
    }

    double get_passive_stress() const
    {
      if (stretch_bar >= 0 && stretch_bar <= 1.)
        return 0.0;
      else if (stretch_bar > 1. && stretch_bar <= 1.25)
        return (2.353844827629192 * pow((stretch_bar - 1),2) + 0.0 * (stretch_bar - 1) + 0.0);
      else if (stretch_bar > 1.25 && stretch_bar <= 1.5)
        return (3.436356700507747 * pow((stretch_bar - 1.25),2) + 1.176922413814596 * (stretch_bar - 1.25) + 0.1471153017268245);
      else if (stretch_bar > 1.5 && stretch_bar <= 1.65)
        return (0.4274082856676522 * pow((stretch_bar - 1.5),2) + 2.8951007640684696 * (stretch_bar - 1.5) + 0.6561181989622077);
      else if (stretch_bar > 1.65)
        return (3.023323249768765 * (stretch_bar - 1.65) + 1.1);
      else
        return 0.0;
    }

    double get_dlength_stress_dstretch() const
    {
      if (stretch_bar >= 0.4 && stretch_bar <= 1.75)
        return  (0.829019797142955 * cos(1.290128342448810 * stretch_bar + 0.629168420414746)
                +1.730615839659180 * cos(5.308969899884336 * stretch_bar + -4.520101562237307)
                +2.213470835640807 * cos(6.744187042136006 * stretch_bar + 1.689155892259429)
                +0.305064635446807 * cos(19.823676877725276 * stretch_bar + -7.386155292116579)
                +1.119254026932584 * cos(8.038287396059996 * stretch_bar + 2.543022326676525)
                +0.058088130602064 * cos(32.237736486095052 * stretch_bar + -6.454098315528945)
                +0.290376594722595 * cos(23.117614057963024 * stretch_bar + -2.643346778503341));
      else
        return 0.0;
    }

    double get_dstrain_rate_stress_dstrain_rate() const
    {
      if (strain_rate_bar < -1.2)
        return  0.0;
      else if (strain_rate_bar >= -1.2 && strain_rate_bar < -0.25)
        return  (3*0.25792669408341773*std::pow(strain_rate_bar+1.2,2)
                            + 2*0.14317485143460784*(strain_rate_bar+1.2) + 0.0);
      else if (strain_rate_bar >= -0.25 && strain_rate_bar < 0.0)
        return  (3*29.825565394304522*std::pow(strain_rate_bar+0.25,2)
                            + 2*-0.9435495605479662*(strain_rate_bar+0.25) + 0.9703687419567255);
      else if (strain_rate_bar >= 0.0 && strain_rate_bar < 0.05)
        return  (3*-3165.6847983144276*std::pow(strain_rate_bar-0.0,2)
                            + 2*186.19612494819665*(strain_rate_bar-0.0) + 6.090887473114851);
      else if (strain_rate_bar >= 0.05 && strain_rate_bar < 0.75)
        return  (3*0.6882206253246714*std::pow(strain_rate_bar-0.05,2)
                            + 2*-1.413963071288272*(strain_rate_bar-0.05) + 0.9678639805763023); 
      else// if (strain_rate_bar >= 0.75)
        return  0.0;
    }

    double get_dpassive_stress_dstretch() const
    {
      if (stretch_bar >= 0 && stretch_bar <= 1.)
        return 0.0;
      else if (stretch_bar > 1. && stretch_bar <= 1.25)
        return (2 * 2.353844827629192 * (stretch_bar - 1));
      else if (stretch_bar > 1.25 && stretch_bar <= 1.5)
        return (2 * 3.436356700507747 * (stretch_bar - 1.25) + 1.176922413814596);
      else if (stretch_bar > 1.5 && stretch_bar <= 1.65)
        return (2 * 0.4274082856676522 * (stretch_bar - 1.5) + 2.8951007640684696);
      else if (stretch_bar > 1.65)
        return (3.023323249768765);
      else
        return 0.0;
    }
  };

  // @sect3{Quadrature point history}

  // As seen in step-44, the <code> PointHistory </code> class offers a method
  // for storing data at the quadrature points. Here each quadrature point
  // holds a pointer to a material description. Among other data, we
  // choose to store the Kirchhoff stress $\boldsymbol{\tau}$ and the tangent
  // $J\mathfrak{c}$ for the quadrature points. It will also be useful to compute
  // forces and energies to store separately the different contributions to the 
  // Kirchhoff stress. Moreover, for the dynamic computation, this is the place
  // where we store "previous" variables.
  template <int dim>
  class PointHistory
  {
  public:
    PointHistory()
      : F_inv(Physics::Elasticity::StandardTensors<dim>::I)
      , tau(SymmetricTensor<2, dim>())
      , tau_vol(SymmetricTensor<2,dim>())
      , tau_iso(SymmetricTensor<2,dim>())
      , tau_iso_muscle_active(SymmetricTensor<2,dim>())
      , tau_iso_muscle_passive(SymmetricTensor<2,dim>())
      , tau_iso_muscle_basematerial(SymmetricTensor<2,dim>())
      , d2Psi_vol_dJ2(0.0)
      , dPsi_vol_dJ(0.0)
      , Jc(SymmetricTensor<4, dim>())
      , displacement(Tensor<1, dim>())
      , displacement_previous(Tensor<1, dim>())
      , grad_displacement(Tensor<2,dim>())
      , velocity_previous((Tensor<1, dim>()))
      , grad_velocity((Tensor<2, dim>()))
      , F_previous(Physics::Elasticity::StandardTensors<dim>::I)
    {}

    virtual ~PointHistory() = default;

    // The first function is used to create a material object and to
    // initialize all tensors correctly: The second one updates the stored
    // values and stresses based on the current deformation measure
    // $\textrm{Grad}\mathbf{u}_{\textrm{n}}$, pressure $\widetilde{p}$, 
    // dilation $\widetilde{J}$ field values, fibre activation $a(t^n)$,
    // and time step size $\delta t$ (recall that the elasticity tensor
    // in the dynamic computation depends on the current time step size).
    void setup_lqp(const Parameters::AllParameters &parameters)
    {
      material =
        std::make_shared<Muscle_Tissues_Three_Field<dim>>(
          parameters.type_of_simulation,
          parameters.max_iso_stress_muscle,
          parameters.kappa_muscle,
          parameters.max_strain_rate,
          parameters.muscle_fibre_orientation_x,
          parameters.muscle_fibre_orientation_y,
          parameters.muscle_fibre_orientation_z,
          parameters.max_iso_stress_basematerial,
          parameters.muscle_basematerial_factor,
          parameters.muscle_basemat_c1,
          parameters.muscle_basemat_c2,
          parameters.muscle_basemat_c3);
      
      update_values(Tensor<1,dim>(),    /*Displacement*/
                    Tensor<2,dim>(),    /*Gradient of displacement*/
                    0.0,                /*Pressure*/
                    1.0,                /*Dilation*/
                    0.0,                /*Activation*/
                    parameters.delta_t  /*Time step size*/);
    }

    // To this end, we calculate the deformation gradient $\mathbf{F}$ from
    // the displacement gradient $\textrm{Grad}\ \mathbf{u}$, i.e.
    // $\mathbf{F}(\mathbf{u}) = \mathbf{I} + \textrm{Grad}\ \mathbf{u}$ and
    // then let the material model associated with this quadrature point
    // update itself. When computing the deformation gradient, we have to take
    // care with which data types we compare the sum $\mathbf{I} +
    // \textrm{Grad}\ \mathbf{u}$: Since $I$ has data type SymmetricTensor,
    // just writing <code>I + Grad_u_n</code> would convert the second
    // argument to a symmetric tensor, perform the sum, and then cast the
    // result to a Tensor (i.e., the type of a possibly nonsymmetric
    // tensor). However, since <code>Grad_u_n</code> is nonsymmetric in
    // general, the conversion to SymmetricTensor will fail. We can avoid this
    // back and forth by converting $I$ to Tensor first, and then performing
    // the addition as between nonsymmetric tensors:
    void update_values(const Tensor<1, dim> &u_n,
                       const Tensor<2, dim> &Grad_u_n,
                       const double          p_tilde,
                       const double          J_tilde,
                       const double          fibre_activation_time,
                       const double          delta_t)
    {
      const Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(Grad_u_n);
      grad_velocity = (F - F_previous) / delta_t;
      material->update_material_data(F, 
                                     p_tilde, 
                                     J_tilde,
                                     fibre_activation_time,
                                     grad_velocity,
                                     delta_t);
      displacement = u_n;
      grad_displacement = Grad_u_n;

      // The material has been updated so we now calculate the Kirchhoff
      // stress $\mathbf{\tau}$, the tangent $J\mathfrak{c}$ and the first and
      // second derivatives of the volumetric free energy.
      //
      // We also store the inverse of the deformation gradient since we
      // frequently use it:
      F_inv                       = invert(F);
      tau                         = material->get_tau();
      tau                         = material->get_tau();
      tau_vol                     = material->get_tau_vol();
      tau_iso                     = material->get_tau_iso();
      tau_iso_muscle_active       = material->get_tau_iso_muscle_active();
      tau_iso_muscle_passive      = material->get_tau_iso_muscle_passive();
      tau_iso_muscle_basematerial = material->get_tau_iso_muscle_basematerial();
      Jc                          = material->get_Jc();
      dPsi_vol_dJ                 = material->get_dPsi_vol_dJ();
      d2Psi_vol_dJ2               = material->get_d2Psi_vol_dJ2();
    }

    // Next, we implement a function that will update the previous variables once
    // the nonlinear step has been completely solved.
    void update_values_timestep(Time &time_object)
    {
      velocity_previous     = (displacement - displacement_previous) / time_object.get_delta_t();
      displacement_previous = displacement;
      // We also update the F_previous variable, needed to compute the current grad_velocity;
      F_previous            = invert(F_inv);
    }

    // We then offer an interface to retrieve certain data.  
    // First, some kinematic variables:
    double get_J_tilde() const
    {
      return material->get_J_tilde();
    }

    double get_det_F() const
    {
      return material->get_det_F();
    }

    double get_stretch() const
    {
      return material->get_stretch();
    }

    double get_stretch_bar() const
    {
      return material->get_stretch_bar();
    }

    double get_strain_rate() const
    {
      return material->get_strain_rate();
    }

    double get_strain_rate_bar() const
    {
      return material->get_strain_rate_bar();
    }

    Tensor<1, dim> get_orientation() const
    {
      return material->get_orientation();
    }

    const Tensor<2, dim> &get_F_inv() const
    {
      return F_inv;
    }

    // ... and in particular for the dynamic case:
    const Tensor<1, dim> &get_displacement() const
    {
      return displacement;
    }

    const Tensor<1, dim> &get_displacement_previous() const
    {
      return displacement_previous;
    }

    const Tensor<2, dim> &get_grad_displacement() const
    {
      return grad_displacement;
    }

    const Tensor<1, dim> &get_velocity_previous() const
    {
      return velocity_previous;
    }

    // Finally, some kinetic variables. These are used in the
    // tangent matrix and residual assembly operations:
    double get_p_tilde() const
    {
      return material->get_p_tilde();
    }

    const SymmetricTensor<2, dim> &get_tau() const
    {
      return tau;
    }

    const SymmetricTensor<2, dim> &get_tau_vol() const
    {
      return tau_vol;
    }

    const SymmetricTensor<2, dim> &get_tau_iso() const
    {
      return tau_iso;
    }

    const SymmetricTensor<2, dim> &get_tau_iso_muscle_active() const
    {
      return tau_iso_muscle_active;
    }

    const SymmetricTensor<2, dim> &get_tau_iso_muscle_passive() const
    {
      return tau_iso_muscle_passive;
    }

    const SymmetricTensor<2, dim> &get_tau_iso_muscle_basematerial() const
    {
      return tau_iso_muscle_basematerial;
    }

    double get_dPsi_vol_dJ() const
    {
      return dPsi_vol_dJ;
    }

    double get_d2Psi_vol_dJ2() const
    {
      return d2Psi_vol_dJ2;
    }

    // Finally, the tangent matrix itself.
    const SymmetricTensor<4, dim> &get_Jc() const
    {
      return Jc;
    }

  // In the spirit of encapsulation, everything that is not needed
  // outside this class is defined as a private member.
  private:
    std::shared_ptr<Muscle_Tissues_Three_Field<dim>> material;

    Tensor<2, dim>          F_inv;
    SymmetricTensor<2, dim> tau;
    SymmetricTensor<2, dim> tau_vol;
    SymmetricTensor<2, dim> tau_iso;
    SymmetricTensor<2, dim> tau_iso_muscle_active;
    SymmetricTensor<2, dim> tau_iso_muscle_passive;
    SymmetricTensor<2, dim> tau_iso_muscle_basematerial;
    double                  d2Psi_vol_dJ2;
    double                  dPsi_vol_dJ;

    SymmetricTensor<4, dim> Jc;

    // Dynamic variables
    Tensor<1, dim> displacement;
    Tensor<1, dim> displacement_previous;
    Tensor<2, dim> grad_displacement; // This variable is needed to compute energies
    Tensor<1, dim> velocity_previous;
    Tensor<2, dim> grad_velocity;     // This variable is updated at each Newton iteration
    Tensor<2, dim> F_previous;        // This variable is updated at each time step
  };

  // @sect3{Incremental displacement class}

  // An important note in this class: because this will be fed into
  // the VectorTools::interpolate_boundary_values function, it MUST
  // inherit Function<dim>. Moreover, to avoid the error message
  // "Dimension 5 not equal to 3" when running in Debug mode, we 
  // must intialize the Function<dim> element using the same number
  // of components that the finite element will have. In the Solid
  // class below, this number is given by n_components = dim + 2. 
  // However, because at this juncture we have not implement Solid 
  // yet, we must hard code this. Note also that, although this might
  // sound like we are about to implement a boundary condition
  // for all three variables (i.e. including pressure and dilation),
  // this discrepancy will be solved when calling component_mask
  // at the time of calling VectorTools::interpolate_boundary_values.
  template <int dim>
  class IncrementalDisplacement : public Function<dim>
  {
  public:
    IncrementalDisplacement(const double u_dir_n, const double u_dir_n_1)
        :
        Function<dim>(dim + 2),
        u_dir_n(u_dir_n),
        u_dir_n_1(u_dir_n_1)
    {}

    virtual double value(const Point<dim> &point,
                          const unsigned int component = 0) const override;
  
  private:
    const double u_dir_n, u_dir_n_1;
  };

  // Note that the implementation below relies on the line of action being
  // the x-axis and that only the x component of the prescribed displacement
  // is nonzero. If any of these conditions change, then the return value
  // must be updated appropriately.
  template <>
  double IncrementalDisplacement<3>::value(const Point<3> &/*point*/,
                                           const unsigned int component) const
  {
    return (component == 0) ? (u_dir_n - u_dir_n_1) : 0.0;
  }

  // @sect3{Dynamic quasi-incompressible finite-strain solid}

  // The Solid class is the central class in that it represents the problem at
  // hand. It follows the usual scheme in that all it really has is a
  // constructor, destructor and a <code>run()</code> function that dispatches
  // all the work to private functions of this class:
  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file,
          const std::string &strain_file,
          const std::string &activation_file);

    void run();

  private:
    // In the private section of this class, we first forward declare a number
    // of objects that are used in parallelizing work using the WorkStream
    // object (see the @ref threads module for more information on this).
    //
    // We declare such structures for the computation of tangent (stiffness)
    // matrix and right hand side vector, and for updating
    // quadrature points:
    struct PerTaskData_K;
    struct ScratchData_K;

    struct PerTaskData_RHS;
    struct ScratchData_RHS;

    struct ScratchData_UQPH;

    // We start the collection of member functions with one that builds the
    // grid:
    void make_grid();

    // Obtain all boundary IDs
    void determine_boundary_ids();

    // Set up the finite element system to be solved:
    void system_setup(PETScWrappers::MPI::BlockVector &solution_delta);

    void determine_component_extractors();

    // Create and update the quadrature points.
    void setup_qph();

    void update_qph_incremental(PETScWrappers::MPI::BlockVector &solution_delta);

    void update_qph_incremental_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_UQPH                                     &scratch);
    
    // Solution retrieval
    PETScWrappers::MPI::BlockVector 
    get_total_solution(const PETScWrappers::MPI::BlockVector &solution_delta) const;

    // In this non-Workstream scenario, we do not need PerTaskData_TIMESTEP and
    // ScratchData_TIMESTEP structures. Moreover, update_timestep calls
    // updates_values_timestep in an embarrasingly parallel way, so there is no need
    // to code a separate update_timestep_one_cell function.
    void update_timestep();

    // Solve for the displacement using a Newton-Raphson method. We break this
    // function into the nonlinear loop and the function that solves the
    // linearized Newton-Raphson step:
    void solve_nonlinear_timestep(PETScWrappers::MPI::BlockVector &solution_delta);

    // Project J=1 onto the dilation finite element space
    void set_initial_dilation();

    // MPI related variables 
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    mutable ConditionalOStream pcout;

    // Several outputs to assess our results. The first function 
    // below will call all the other ones.
    void output_results();
    void output_vtk() const;
    
    // Parameters from .prm file
    Parameters::AllParameters parameters;

    // ...the volume of the reference configuration...
    double vol_reference;

    // ...and description of the geometry on which the problem is solved:
    parallel::distributed::Triangulation<dim> triangulation;
    std::vector<unsigned int> list_of_boundary_ids;

    // Also, keep track of the current time and the time spent evaluating
    // certain functions
    Time                 time;
    mutable TimerOutput  timer;

    // Create pulling profile
    TabularFunction u_dir;

    // Create activation profile
    TabularFunction activation_function;

    // A storage object for quadrature point information. As opposed to
    // step-18, deal.II's native quadrature point data manager is employed
    // here.
    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim>> quadrature_point_history;

    // A description of the finite-element system including the displacement
    // polynomial degree, the degree-of-freedom handler, number of DoFs per
    // cell and the extractor objects used to retrieve information from the
    // solution vectors:
    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;
    const FEValuesExtractors::Scalar p_fe;
    const FEValuesExtractors::Scalar J_fe;

    // Description of how the block-system is arranged. There are 3 blocks,
    // the first contains a vector DOF $\mathbf{u}$ while the other two
    // describe scalar DOFs, $\widetilde{p}$ and $\widetilde{J}$.
    static const unsigned int n_blocks          = 3;
    static const unsigned int n_components      = dim + 2;
    static const unsigned int first_u_component = 0;
    static const unsigned int p_component       = dim;
    static const unsigned int J_component       = dim + 1;

    enum
    {
      u_dof = 0,
      p_dof = 1,
      J_dof = 2
    };

    std::vector<types::global_dof_index> dofs_per_block;
    std::vector<types::global_dof_index> element_indices_u;
    std::vector<types::global_dof_index> element_indices_p;
    std::vector<types::global_dof_index> element_indices_J;

    // Rules for Gauss-quadrature on both the cell and faces. The number of
    // quadrature points on both cells and faces is recorded.
    const QGauss<dim>     qf_cell;
    // const QGauss<dim - 1> qf_face;
    const unsigned int    n_q_points;
    // const unsigned int    n_q_points_f;

    // More MPI related variables
    std::vector<unsigned int> block_component;
    std::vector<IndexSet> all_locally_owned_dofs;
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    std::vector<IndexSet> locally_owned_partitioning;
    std::vector<IndexSet> locally_relevant_partitioning;

    // Objects that store the converged solution and right-hand side vectors,
    // as well as the tangent matrix. There is an AffineConstraints object used
    // to keep track of constraints.
    AffineConstraints<double>             constraints;
    PETScWrappers::MPI::BlockSparseMatrix tangent_matrix;
    PETScWrappers::MPI::BlockVector       system_rhs;
    PETScWrappers::MPI::BlockVector       solution_n_relevant;

    // Store the outputs in a separate folder
    char save_dir[80];
  };

  // @sect3{Implementation of the <code>Solid</code> class}

  // @sect4{Public interface}

  // We initialize the Solid class using data extracted from the parameter file.
  template <int dim>
  Solid<dim>::Solid(const std::string &input_file,
                    const std::string &strain_file,
                    const std::string &activation_file)
    : mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , pcout(std::cout, this_mpi_process == 0)
    , parameters(input_file)
    , vol_reference(0.)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
    , time(parameters.end_time, parameters.delta_t)
    , timer(mpi_communicator,
            pcout,
            TimerOutput::summary,
            TimerOutput::wall_times)
    , u_dir(strain_file)
    , activation_function(activation_file)
    , degree(parameters.poly_degree)
    ,
    // The Finite Element System is composed of dim continuous displacement
    // DOFs, and discontinuous pressure and dilatation DOFs. In an attempt to
    // satisfy the Babuska-Brezzi or LBB stability conditions (see Hughes
    // (2000)), we setup a $Q_n \times DGPM_{n-1} \times DGPM_{n-1}$
    // system. $Q_2 \times DGPM_1 \times DGPM_1$ elements satisfy this
    // condition, while $Q_1 \times DGPM_0 \times DGPM_0$ elements do
    // not. However, it has been shown that the latter demonstrate good
    // convergence characteristics nonetheless.
    fe(FE_Q<dim>(parameters.poly_degree),
       dim, // displacement
       FE_DGPMonomial<dim>(parameters.poly_degree - 1),
       1, // pressure
       FE_DGPMonomial<dim>(parameters.poly_degree - 1),
       1) // dilatation
    , dof_handler(triangulation)
    , dofs_per_cell(fe.n_dofs_per_cell())
    , u_fe(first_u_component)
    , p_fe(p_component)
    , J_fe(J_component)
    , dofs_per_block(n_blocks)
    , qf_cell(parameters.quad_order)
    , n_q_points(qf_cell.size())
  {
    Assert(dim == 2 || dim == 3,
           ExcMessage("This problem only works in 2 or 3 space dimensions."));
    determine_component_extractors();
    (void)strain_file;
    (void)activation_file;

    // Initialize save_dir first with the current timestamp.
    if (this_mpi_process == 0)
    {
      std::chrono::system_clock::time_point time_now;
      time_t time_conv;
      struct tm* timeinfo;

      time_now = std::chrono::system_clock::now();
      time_conv = std::chrono::system_clock::to_time_t(time_now);
      timeinfo = localtime(&time_conv);
      strftime(save_dir,80,"%Y%m%d_%H%M%S",timeinfo);

      // Then, we append to the timestamp some letters to
      // quickly visualize the type of simulation that we just
      // performed:

      // Append _Q or _D depending on the type of simulation
      if (parameters.type_of_simulation == "quasi-static")
        strcat(save_dir, "_Q");
      else if (parameters.type_of_simulation == "dynamic")
        strcat(save_dir, "_D");

      // Append nonlinear solver info
      if (parameters.type_nonlinear_solver == "classicNewton")
        strcat(save_dir, "C");
      else if (parameters.type_nonlinear_solver == "acceleratedNewton")
        strcat(save_dir, "A");

      // Append linear solver info
      if (parameters.type_lin == "CG")
        strcat(save_dir, "C");
      else if (parameters.type_lin == "GMRES")
        strcat(save_dir, "G");
      else if (parameters.type_lin == "Direct")
        strcat(save_dir, "D");
      
      // Append preconditioner info
      if (parameters.type_lin == "Direct")
        strcat(save_dir, "X");
      else if (parameters.preconditioner_type == "ssor" && parameters.type_lin != "Direct")
        strcat(save_dir, "S");
      else if (parameters.preconditioner_type == "jacobi" && parameters.type_lin != "Direct")
        strcat(save_dir, "J");

      // Static condensation?
      if (parameters.use_static_condensation)
        strcat(save_dir, "T");
      else
        strcat(save_dir, "F");
    }

    // Broadcast save_dir variable to the other processes
    MPI_Bcast(&save_dir, 80, MPI_CHAR, 0, MPI_COMM_WORLD);
  }

  // Similarly to step-44, we start the function with preprocessing, 
  // setting the initial dilatation values, and then output the initial 
  // grid and the parameters used before starting the simulation in itself.
  //
  // Care must be taken (or at least some thought given) when imposing the
  // constraint $\widetilde{J}=1$ on the initial solution field. The constraint
  // corresponds to the determinant of the deformation gradient in the
  // undeformed configuration, which is the identity tensor. We use
  // FE_DGPMonomial bases to interpolate the dilatation field, thus we can't
  // simply set the corresponding dof to unity as they correspond to the
  // monomial coefficients. Thus we use the VectorTools::project function to do
  // the work for us. The VectorTools::project function requires an argument
  // indicating the hanging node constraints. We have none in this program
  // So we have to create a constraint object. In its original state, constraint
  // objects are unsorted, and have to be sorted (using the
  // AffineConstraints::close function) before they can be used. Have a look at
  // step-21 for more information. We only need to enforce the initial condition
  // on the dilatation. In order to do this, we make use of a
  // ComponentSelectFunction which acts as a mask and sets the J_component of
  // n_components to 1. This is exactly what we want. Have a look at its usage
  // in step-20 for more information.
  template <int dim>
  void Solid<dim>::run()
  {
    pcout << "--------------------------------------------------------" << "\n"
          << "                                                        " << "\n"
          << "             F L E X O D E A L  ( L I T E )             " << "\n"
          << "                                                        " << "\n"
          << "--------------------------------------------------------" << "\n" << std::endl;

    pcout << "Running in parallel with "
          << n_mpi_processes
          << " processes.\n" << std::endl;

    // Create director to store outputs, store parameters file
    if (this_mpi_process == 0)
    {
      // Create directory to store all the outputs
      if (mkdir(save_dir, 0777) == -1)
        std::cerr << "Error :  " << strerror(errno) << std::endl;

      std::ostringstream prm_output_filename;
      prm_output_filename << save_dir << "/parameters.prm";
      std::ofstream out(prm_output_filename.str().c_str());
      parameters.prm.print_parameters(out, 
                                      ParameterHandler::PRM | 
                                      ParameterHandler::KeepDeclarationOrder);
    }

    // Create grid for the problem
    make_grid();

    // Obtain all boundary IDs
    determine_boundary_ids();

    // Setup system, initialize solution_delta and solution_n_relevant
    PETScWrappers::MPI::BlockVector solution_delta;
    system_setup(solution_delta);

    // Set initial dilation J = 1. In a non-PETSc/Trilinos code, this
    // can be done using the VectorTools::project function, but because
    // this function does not seem to be available yet for distributed
    // vectors, we reimplement this function from scratch.
    set_initial_dilation();

    // Output initial solution
    output_results();

    time.increment();
    // We then declare the incremental solution update $\varDelta
    // \mathbf{\Xi} \dealcoloneq \{\varDelta \mathbf{u},\varDelta \widetilde{p},
    // \varDelta \widetilde{J} \}$ and start the loop over the time domain.
    //while (time.current() < time.end())
    {
      // ...solve the current time step and update total solution vector
      // $\mathbf{\Xi}_{\textrm{n}} = \mathbf{\Xi}_{\textrm{n-1}} +
      // \varDelta \mathbf{\Xi}$...
      solve_nonlinear_timestep(solution_delta);
      {
        // We have to use a non-ghosted version of solution_n_relevant to
        // do the addition
        PETScWrappers::MPI::BlockVector tmp(locally_owned_partitioning,
                                            mpi_communicator);
        tmp = solution_n_relevant;
        tmp += solution_delta;
        solution_n_relevant = tmp;
      }
      // ...and output the results (including VTU files) before moving on 
      // happily to the next time step:
      output_results();

      // If our computation is dynamic (rather than quasi-static),
      // then we have to update the "previous" variables. 
      if (parameters.type_of_simulation == "dynamic")
        update_timestep();

      time.increment();

      // Reset the solution_delta object for the next timestep
      solution_delta = 0.0;
    }

  }

  // And finally we define the structures to assist with updating the quadrature
  // point information.
  // The ScratchData object will be used to store an alias for the solution
  // vector so that we don't have to copy this large data structure. We then
  // define a number of vectors to extract the solution values and gradients at
  // the quadrature points.
  template <int dim>
  struct Solid<dim>::ScratchData_UQPH
  {
    const PETScWrappers::MPI::BlockVector &solution_total;

    std::vector<Tensor<1, dim>> solution_values_u_total;
    std::vector<Tensor<2, dim>> solution_grads_u_total;
    std::vector<double>         solution_values_p_total;
    std::vector<double>         solution_values_J_total;

    FEValues<dim> fe_values;

    ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
                     const QGauss<dim>        &qf_cell,
                     const UpdateFlags         uf_cell,
                     const PETScWrappers::MPI::BlockVector &solution_total)
      : solution_total(solution_total)
      , solution_values_u_total(qf_cell.size())
      , solution_grads_u_total(qf_cell.size())
      , solution_values_p_total(qf_cell.size())
      , solution_values_J_total(qf_cell.size())
      , fe_values(fe_cell, qf_cell, uf_cell)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      : solution_total(rhs.solution_total)
      , solution_values_u_total(rhs.solution_values_u_total)
      , solution_grads_u_total(rhs.solution_grads_u_total)
      , solution_values_p_total(rhs.solution_values_p_total)
      , solution_values_J_total(rhs.solution_values_J_total)
      , fe_values(rhs.fe_values.get_fe(),
                  rhs.fe_values.get_quadrature(),
                  rhs.fe_values.get_update_flags())
    {}

    void reset()
    {
      const unsigned int n_q_points = solution_grads_u_total.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          solution_values_u_total[q] = 0.0;
          solution_grads_u_total[q]  = 0.0;
          solution_values_p_total[q] = 0.0;
          solution_values_J_total[q] = 0.0;
        }
    }
  };

  template <int dim>
  void Solid<dim>::update_timestep()
  {
    TimerOutput::Scope t(timer, "Update time stepping");
    pcout << "\n" << "-------- Updating time stepping --------" << "\n" << std::flush;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const std::vector<std::shared_ptr<PointHistory<dim>>>
        lqph = quadrature_point_history.get_data(cell);
    
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->update_values_timestep(time);
      }
  }

  // @sect4{Solid::get_total_solution}

  // This function provides the total solution, which is valid at any Newton step.
  // This is required as, to reduce computational error, the total solution is
  // only updated at the end of the timestep.

  template <int dim>
  PETScWrappers::MPI::BlockVector 
  Solid<dim>::get_total_solution(const PETScWrappers::MPI::BlockVector &solution_delta) const
  {
    // The output solution_total should be GHOSTED so we can compute function values and
    // gradients inside ScratchData_UQPH. However, we cannot add this vector directly
    // to solution_delta because this operation required NON-GHOSTED vectors. Therefore,
    // we have to use a temporal variable to perform the addition.
    
    PETScWrappers::MPI::BlockVector solution_total(locally_owned_partitioning,
                                                   locally_relevant_partitioning,
                                                   mpi_communicator);

    PETScWrappers::MPI::BlockVector tmp(locally_owned_partitioning,
                                        mpi_communicator);
    
    tmp = solution_n_relevant; // solution_n_relevant is ghosted
    tmp += solution_delta;     // tmp and solution_delta are not ghosted, they can be added
    solution_total = tmp;      // Assign the values of tmp to the ghosted vector of interest
    
    return solution_total;
  }

  template <int dim>
  void Solid<dim>::update_qph_incremental(PETScWrappers::MPI::BlockVector &solution_delta)
  {
    TimerOutput::Scope t(timer, "Update QPH data");
    //pcout << " UQPH " << std::flush;

    const PETScWrappers::MPI::BlockVector solution_total(get_total_solution(solution_delta));
    const UpdateFlags uf_UQPH(update_values | update_gradients);
    ScratchData_UQPH  scratch_data_UQPH(fe, qf_cell, uf_UQPH, solution_total);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        update_qph_incremental_one_cell(cell, scratch_data_UQPH);
  }

  template <int dim>
  void Solid<dim>::update_qph_incremental_one_cell(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData_UQPH &scratch)
  {
    const std::vector<std::shared_ptr<PointHistory<dim>>> lqph =
      quadrature_point_history.get_data(cell);
    Assert(lqph.size() == n_q_points, ExcInternalError());

    Assert(scratch.solution_values_u_total.size() == n_q_points,
           ExcInternalError());
    Assert(scratch.solution_grads_u_total.size() == n_q_points,
           ExcInternalError());
    Assert(scratch.solution_values_p_total.size() == n_q_points,
           ExcInternalError());
    Assert(scratch.solution_values_J_total.size() == n_q_points,
           ExcInternalError());

    scratch.reset();

    // We first need to find the values and gradients at quadrature points
    // inside the current cell and then we update each local QP using the
    // displacement gradient and total pressure and dilatation solution
    // values:
    scratch.fe_values.reinit(cell);
    scratch.fe_values[u_fe].get_function_values(
      scratch.solution_total, scratch.solution_values_u_total);
    scratch.fe_values[u_fe].get_function_gradients(
      scratch.solution_total, scratch.solution_grads_u_total);
    scratch.fe_values[p_fe].get_function_values(
      scratch.solution_total, scratch.solution_values_p_total);
    scratch.fe_values[J_fe].get_function_values(
      scratch.solution_total, scratch.solution_values_J_total);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      lqph[q_point]->update_values(scratch.solution_values_u_total[q_point],
                                   scratch.solution_grads_u_total[q_point],
                                   scratch.solution_values_p_total[q_point],
                                   scratch.solution_values_J_total[q_point],
                                   activation_function(time.current()),
                                   time.get_delta_t());
  }



  // @sect4{Solid::make_grid}

  // On to the first of the private member functions. Here we create the
  // triangulation of the domain, for which we choose the scaled cube with each
  // face given a boundary ID number.
  //
  // We then determine the volume of the reference configuration and print it
  // for comparison:
  template <int dim>
  void Solid<dim>::make_grid()
  {
    GridGenerator::hyper_rectangle(
      triangulation,
      (dim == 3 ? Point<dim>(0.0, 0.0, 0.0) : Point<dim>(0.0, 0.0)),
      (dim == 3 ? Point<dim>(parameters.length, parameters.width, parameters.height) : Point<dim>(parameters.length, parameters.height)),
      true);
    GridTools::scale(parameters.scale, triangulation);
    triangulation.refine_global(std::max(1U, parameters.global_refinement));

    vol_reference = GridTools::volume(triangulation);
    pcout << "Geometry:"
          << "\n\t Reference volume: " << vol_reference << " m^3"
          << "\n\t Mass:             " << vol_reference * parameters.muscle_density << " kg"
          << "\n" << std::endl;

    // then, we output the grid used for future reference.
    if (this_mpi_process == 0)
    {
      std::ostringstream filename;
      filename << save_dir << "/grid-" << dim << "d"<< ".msh";
      GridOut           grid_out;
      GridOutFlags::Msh write_flags;
      write_flags.write_faces = true;
      grid_out.set_flags(write_flags);
      std::ofstream output(filename.str().c_str());
      grid_out.write_msh(triangulation, output);
    }
  }
  
  // @sect4{Solid::determine_boundary_ids}

  // This is a simple function to retrieve all the boundary IDs set in the mesh.
  // While this might be simple for a block geometry (the IDs are {0,1,2,3,4,5}),
  // it might not be the case for other types of meshes. This will allow us to
  // output forces for each boundary ID.
  template <int dim>
  void Solid<dim>::determine_boundary_ids()
  {
    // Note that, because the triangulation was defined as a "parallel::shared" 
    // object, rather than "parallel::distributed", all processes should have access 
    // to the entire mesh, and therefore, the output (list_of_boundary_ids) should be
    // the same.

    // Loop over all boundary cells and boundary faces
    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->at_boundary())
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
          {
            const unsigned int id = cell->face(face)->boundary_id();
            const bool id_exists_in_list = std::find(std::begin(list_of_boundary_ids), 
                                                     std::end(list_of_boundary_ids), id) 
                                                     != std::end(list_of_boundary_ids);
            if (!id_exists_in_list)
              list_of_boundary_ids.push_back(id);
          }

    // Sort the array
    std::sort(list_of_boundary_ids.begin(), list_of_boundary_ids.end());
  }

  template <int dim>
  void Solid<dim>::system_setup(PETScWrappers::MPI::BlockVector &solution_delta)
  {
    TimerOutput::Scope t(timer, "Setup system");
    pcout << "Setting up linear system structure..." << std::endl;

    std::fill_n(std::back_inserter(block_component), n_components, u_dof); // Displacement
    block_component[p_component] = p_dof;             // Pressure
    block_component[J_component] = J_dof;             // Dilatation

    // The DOF handler is then initialised and we renumber the grid in
    // an efficient manner. We also record the number of DOFs per block.
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);
    DoFRenumbering::component_wise(dof_handler, block_component);

    // Count DoFs in each block
    dofs_per_block.clear();
    dofs_per_block.resize(n_blocks);
    dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const unsigned int n_u = dofs_per_block[u_dof],
                        n_p = dofs_per_block[p_dof],
                        n_J = dofs_per_block[J_dof];

    pcout << "\tNumber of degrees of freedom per block: "
          << "[n_u, n_p, n_J] = ["
          << n_u << ", "
          << n_p << ", "
          << n_J << "]"
          << std::endl;

    // We now define what locally_owned_partitioning and 
    // locally_relevant_partitioning are. We follow step-55
    // for this.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_owned_partitioning.resize(n_blocks);
    locally_owned_partitioning[u_dof] = locally_owned_dofs.get_view(0, n_u);
    locally_owned_partitioning[p_dof] = locally_owned_dofs.get_view(n_u, n_u+n_p);
    locally_owned_partitioning[J_dof] = locally_owned_dofs.get_view(n_u+n_p, n_u+n_p+n_J);

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    locally_relevant_partitioning.resize(n_blocks);
    locally_relevant_partitioning[u_dof] = locally_relevant_dofs.get_view(0,n_u);
    locally_relevant_partitioning[p_dof] = locally_relevant_dofs.get_view(n_u, n_u+n_p);
    locally_relevant_partitioning[J_dof] = locally_relevant_dofs.get_view(n_u+n_p, n_u+n_p+n_J);

    // Setup the sparsity pattern and tangent matrix
    {
      tangent_matrix.clear();

      // We optimise the sparsity pattern to reflect the particular
      // structure of the system matrix and prevent unnecessary data 
      // creation for the right-diagonal block components.
      Table<2, DoFTools::Coupling> coupling(n_components, n_components);
      for (unsigned int ii = 0; ii < n_components; ++ii)
          for (unsigned int jj = 0; jj < n_components; ++jj)
          {
              if ((   (ii <  p_component) && (jj == J_component))
                  || ((ii == J_component) && (jj < p_component))
                  || ((ii == p_component) && (jj == p_component)  ))
                  coupling[ii][jj] = DoFTools::none;
              else
                  coupling[ii][jj] = DoFTools::always;
          }
      
      BlockDynamicSparsityPattern dsp(locally_relevant_partitioning);
      
      DoFTools::make_sparsity_pattern(dof_handler, 
                                      coupling,
                                      dsp,
                                      constraints,
                                      false);
      
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dof_handler.locally_owned_dofs(),
        mpi_communicator,
        locally_relevant_dofs);
      
      tangent_matrix.reinit(locally_owned_partitioning,
                            dsp,
                            mpi_communicator);
    }

    // Finally construct the block vectors with the right sizes.
    // The solution vector we seek does not only store
    // elements we own, but also ghost entries; on the other hand, the right
    // hand side vector only needs to have the entries the current processor
    // owns since all we will ever do is write into it, never read from it on
    // locally owned cells (of course the linear solvers will read from it,
    // but they do not care about the geometric location of degrees of
    // freedom).
    system_rhs.reinit(locally_owned_partitioning, 
                      mpi_communicator);
    solution_n_relevant.reinit(locally_owned_partitioning,
                               locally_relevant_partitioning, 
                               mpi_communicator);
    solution_delta.reinit(locally_owned_partitioning, 
                          mpi_communicator);

    // Setup quadrature point history
    setup_qph();

    pcout << "Quadrature point data has been set up:"
          << "\n\t Linear solver: " << parameters.type_lin 
          << "\n\t Non-linear solver: " << parameters.type_nonlinear_solver << "\n"
          << std::endl;

    pcout << "Running " << parameters.type_of_simulation << " muscle contraction" 
          << std::endl;
  }

  template <int dim>
  void Solid<dim>::determine_component_extractors()
  {
    element_indices_u.clear();
    element_indices_p.clear();
    element_indices_J.clear();

    for (unsigned int k = 0; k < fe.n_dofs_per_cell(); ++k)
      {
        const unsigned int k_group = fe.system_to_base_index(k).first.first;
        if (k_group == u_dof)
          element_indices_u.push_back(k);
        else if (k_group == p_dof)
          element_indices_p.push_back(k);
        else if (k_group == J_dof)
          element_indices_J.push_back(k);
        else
          {
            Assert(k_group <= J_dof, ExcInternalError());
          }
      }
  }

  template <int dim>
  void Solid<dim>::setup_qph()
  {
    pcout << "\nSetting up quadrature point data...\n" << std::endl;

    {
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>
        f_cell (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler.begin_active()),
        f_endc (IteratorFilters::SubdomainEqualTo(this_mpi_process), dof_handler.end());
      
      quadrature_point_history.initialize(f_cell, f_endc, n_q_points);
    }

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        Assert(cell->subdomain_id()==this_mpi_process, ExcInternalError());
        const std::vector<std::shared_ptr<PointHistory<dim>>>
          lqph = quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters);
      }
  }

  // @sect4{Solid::set_initial_dilation}

  // The VectorTools::project() is not available in deal.II v9.5.0 for an MPI-based
  // code, so we have to implement it by ourselves. Recall that the main objective
  // is to initialize the dilation as J = 1 by projecting this function onto the
  // dilation finite element space.
  template <int dim>
  void Solid<dim>::set_initial_dilation()
  {
    TimerOutput::Scope t(timer, "Setup initial dilation");
    pcout << "\nSetting up initial dilation...\n" << std::endl;
    DoFHandler<dim> dof_handler_J(triangulation);
    FE_DGPMonomial<dim> fe_J(parameters.poly_degree - 1);

    IndexSet                      locally_owned_dofs_J;
    IndexSet                      locally_relevant_dofs_J;
    AffineConstraints             constraints_J;
    PETScWrappers::MPI::SparseMatrix   mass_matrix;
    PETScWrappers::MPI::Vector    load_vector;
    PETScWrappers::MPI::Vector    J_local; 

    // Setup system
    dof_handler_J.distribute_dofs(fe_J);
    locally_owned_dofs_J = dof_handler_J.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler_J, locally_relevant_dofs_J);

    J_local.reinit(locally_owned_dofs_J, locally_relevant_dofs_J, mpi_communicator);
    load_vector.reinit(locally_owned_dofs_J, mpi_communicator);

    constraints_J.clear();
    constraints_J.reinit(locally_relevant_dofs_J);
    constraints_J.close();

    DynamicSparsityPattern dsp_J(locally_relevant_dofs_J);
    DoFTools::make_sparsity_pattern(
      dof_handler_J, dsp_J, constraints_J, false);
    SparsityTools::distribute_sparsity_pattern(
      dsp_J,
      dof_handler_J.locally_owned_dofs(),
      mpi_communicator,
      locally_relevant_dofs_J);
    mass_matrix.reinit(
      locally_owned_dofs_J, locally_owned_dofs_J, dsp_J, mpi_communicator);

    // Assemble system
    const QGauss<dim> quad_formula(parameters.quad_order);
    FEValues<dim> fe_values_J(
      fe_J, quad_formula, 
      update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell_J = fe_J.dofs_per_cell;
    const unsigned int n_q_points_J = quad_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell_J, dofs_per_cell_J);
    Vector<double> cell_rhs(dofs_per_cell_J);
    std::vector<types::global_dof_index> local_dof_indices_J(dofs_per_cell_J);

    for (const auto &cell : dof_handler_J.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values_J.reinit(cell);

        for (unsigned int q_point = 0; q_point < n_q_points_J; ++q_point)
          for (unsigned int i = 0; i < dofs_per_cell_J; ++i)
          {
            cell_rhs(i) += (1 * 
                            fe_values_J.shape_value(i,q_point) * 
                            fe_values_J.JxW(q_point));
            
            for (unsigned int j = 0; j < dofs_per_cell_J; ++j)
              cell_matrix(i,j) += (fe_values_J.shape_value(i, q_point) *
                                   fe_values_J.shape_value(j, q_point) *
                                   fe_values_J.JxW(q_point));
          }

        cell->get_dof_indices(local_dof_indices_J);
        constraints_J.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices_J, mass_matrix, load_vector);
      }

    // Notice that the assembling above is just a local operation. So, to
    // form the "global" linear system, a synchronization between all
    // processors is needed. This could be done by invoking the function
    // compress(). See @ref GlossCompress  "Compressing distributed objects"
    // for more information on what is compress() designed to do. 
    mass_matrix.compress(VectorOperation::add);
    load_vector.compress(VectorOperation::add);

    // Solve
    PETScWrappers::MPI::Vector J_distributed(locally_owned_dofs_J, mpi_communicator);
    SolverControl solver_control(dof_handler_J.n_dofs(), 1e-12);
    PETScWrappers::SolverCG solver(solver_control);

    PETScWrappers::PreconditionJacobi preconditioner;
    preconditioner.initialize(mass_matrix);
    solver.solve(mass_matrix, J_distributed, load_vector, preconditioner);
    constraints_J.distribute(J_distributed);

    // We have computed the J=1 in the finite element space. We now transfer
    // this quantity to the solution_n_relevant vector.
    solution_n_relevant.block(J_dof) = J_distributed;

    // We destroy the dof_handler_J object and leave the subsection.
    dof_handler_J.clear();
  }

  template <int dim>
  void Solid<dim>::solve_nonlinear_timestep(
    PETScWrappers::MPI::BlockVector &solution_delta)
  {
    pcout << std::endl
          << "Timestep " << time.get_timestep() << " @ " << time.current()
          << "s" << std::endl;

    pcout << "Current activation: " << activation_function(time.current()) * 100 << "%\n"
          << "Current strain:     " << u_dir(time.current())
          << std::endl;

    //BlockVector<double> newton_update(dofs_per_block);

    //error_residual.reset();
    //error_residual_0.reset();
    //error_residual_norm.reset();
    //error_update.reset();
    //error_update_0.reset();
    //error_update_norm.reset();

    //print_conv_header();

    update_qph_incremental(solution_delta);
  }

  // @sect4{Solid::output_results}
  // The output_results function looks a bit different from other tutorials.
  // This is because not only we're interested in visualizing Paraview files
  // but also traces (time series) of other quantities. The Paraview files
  // are exported in the output_vtk function.
  template <int dim>
  void Solid<dim>::output_results()
  {
    TimerOutput::Scope t(timer, "Output results");
    output_vtk();
  }

  template <int dim>
  void Solid<dim>::output_vtk() const
  {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

      std::vector<std::string> solution_name(dim, "displacement");
      solution_name.emplace_back("pressure");
      solution_name.emplace_back("dilatation");

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution_n_relevant,
                               solution_name,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);
      

      // As a last piece of data, let us also add the partitioning of the domain
      // into subdomains associated with the processors if this is a parallel
      // job. This works in the exact same way as in the step-17 program:
      std::vector<types::subdomain_id> partition_int(
        triangulation.n_active_cells());
      GridTools::get_subdomain_association(triangulation, partition_int);
      const Vector<double> partitioning(partition_int.begin(),
                                        partition_int.end());
      data_out.add_data_vector(partitioning, "partitioning");
      
      // Since we are dealing with a large deformation problem, it would be nice
      // to display the result on a displaced grid!  The MappingQEulerian class
      // linked with the DataOut class provides an interface through which this
      // can be achieved without physically moving the grid points in the
      // Triangulation object ourselves.  We first need to copy the solution to
      // a temporary vector and then create the Eulerian mapping. We also
      // specify the polynomial degree to the DataOut object in order to produce
      // a more refined output data set when higher order polynomials are used.
      MappingQEulerian<dim, PETScWrappers::MPI::BlockVector>
      q_mapping(degree, dof_handler, solution_n_relevant);

      data_out.build_patches(q_mapping, degree);

      std::string str_save_dir(save_dir);
      str_save_dir += "/";

      const std::string pvtu_filename = data_out.write_vtu_with_pvtu_record(
        str_save_dir, "solution-3d", time.get_timestep(), mpi_communicator, 4);
      
      if (this_mpi_process == 0)
      {
        static std::vector<std::pair<double, std::string>> times_and_names;
        times_and_names.emplace_back(time.current(), pvtu_filename);
        std::ofstream pvd_output(str_save_dir + "solution-3d.pvd");
        DataOutBase::write_pvd_record(pvd_output, times_and_names);
      }
  }

} // End of namespace Flexodeal


// @sect3{Main function}
// Lastly we provide the main driver function. Here, we have the option
// of using different parameter, strain, and activation files without the
// need to recompile the code.
int main(int argc, char* argv[])
{
  using namespace Flexodeal;
  
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  try
    {
      // The program only works for dim = 3. 
      // Maybe one day will also work for dim = 2 ...
      const unsigned int dim = 3;
      std::string parameters_file, strain_file, activation_file;
      
      if (argc == 1)
      {
        // If no extra arguments are given (such as when calling "make run" 
        // or just "./dynamic-muscle"), then the following files are
        // considered by default:
        parameters_file = "parameters.prm";
        strain_file     = "control_points_strain.dat";
        activation_file = "control_points_activation.dat";
        
        // Caution must be taken when using files that are different from
        // these default values. Whatever name you use for these files,
        // the order must be preserved. 
        //
        // Remember: public service announcement, PSA (parameters, strain, 
        // activation)! Credits to Kshitij Patil for the acronym :)
      } 
      else
      {
        parameters_file = argv[1];
        strain_file     = argv[2];
        activation_file = argv[3];
      }

      Solid<dim>  solid(parameters_file, strain_file, activation_file);
      solid.run();
    }
  catch (std::exception &exc)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
      }
      return 1;
    }
  catch (...)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
      }
    }

  return 0;
}

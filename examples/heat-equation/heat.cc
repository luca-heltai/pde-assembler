#include <pidomus.h>
#include "heat_interface.h"

int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);


  const int dim = 2;
  const int spacedim = 2;

  // for serial version using a direct solver use uncomment these two
  // lines
   HeatEquation<dim,spacedim,LADealII> problem;
   piDoMUS<dim,spacedim,LADealII> solver ("pidomus",problem);

  // for parallel version using an iterative solver uncomment these
  // two lines
//  HeatEquation<dim,spacedim,LATrilinos> problem;
//  piDoMUS<dim,spacedim,LATrilinos> solver ("pidomus",problem);

  IMEXStepper<typename LADealII::VectorType> imex{"Outer imex", MPI_COMM_WORLD};



  ParameterAcceptor::initialize("heat.prm", "used_parameters.prm");


  solver.current_alpha = imex.get_alpha();
  imex.create_new_vector = solver.lambdas.create_new_vector;
  imex.residual = solver.lambdas.residual;
  imex.setup_jacobian = solver.lambdas.setup_jacobian;
  imex.solver_should_restart = solver.lambdas.solver_should_restart;
  imex.solve_jacobian_system = solver.lambdas.solve_jacobian_system;
  imex.output_step = solver.lambdas.output_step;
  imex.get_lumped_mass_matrix = solver.lambdas.get_lumped_mass_matrix;
  imex.jacobian_vmult = solver.lambdas.jacobian_vmult;


//  solver.run ();

solver.init();

  imex.solve_dae(solver.solution, solver.solution_dot);



  return 0;
}

#ifndef QUASI_STATIC_PROBLEM_H
#define QUASI_STATIC_PROBLEM_H

#include "lac/lac_type.h"
#include "pde_base_interface.h"
#include "pde_handler.h"
#include "pde_handler_access.h"
#include "pidomus_signals.h"

#include <deal2lkit/error_handler.h>
#include "deal2lkit/parsed_solver.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/mpi.h>


using namespace deal2lkit;
using namespace dealii;

template<int dim, int spacedim, typename LAC>
class QuasiStaticProblem : public dealii::ParameterAcceptor, public PDEHandlerAccess<dim,spacedim,LAC>
{
public:
  QuasiStaticProblem(const std::string &name,
                     PDEBaseInterface<dim,spacedim,LAC> &interface,
                     const MPI_Comm &comm=MPI_COMM_WORLD);

  /**
   * Call the corresponding init function of the pde_handler.
   */
  void init();

  /**
   * Actually perform quasi static loop.
   */
  void run();

private:
  /**
   * The mpi communicator.
   */
  const MPI_Comm &comm;

  /**
   * Interface to the pde object.
   */
  PDEBaseInterface<dim,spacedim,LAC> &interface;

  /**
   * The PDEHandler object.
   */
  PDEHandler<dim,spacedim,LAC> pde;

  /**
   * Number of cycles to perform.
   */
  unsigned int n_cycles = 1;

  /**
   * Pseudo time step, used for quasi static computations. If set to zero, then
   * only perform one simulation and then exit, otherwise perform computations
   * up to the `final_time`, using this `time_step`.
   */
  double time_step = 0;

  /**
   * Starting time.
   */
  double start_time = 0;

  /**
   * Ending time.
   */
  double final_time = 1.0;

  /**
   * Error handler.
   */
  ErrorHandler<1> eh;

  /**
   * Exact solution.
   */
  ParsedFunction<spacedim>        exact_solution;

  /**
   * Parsed solver.
   */
  ParsedSolver<typename LAC::VectorType> solver;
};

#endif // QUASISTATICPROBLEM_H

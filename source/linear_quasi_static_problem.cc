#include "linear_quasi_static_problem.h"
#include "pidomus_macros.h"

#include <deal.II/lac/sparse_direct.h>

#include <deal2lkit/utilities.h>

template<int dim, int spacedim, typename LAC>
LinearQuasiStaticProblem<dim,spacedim,LAC>::LinearQuasiStaticProblem(const std::string &name,
                                                         PDEBaseInterface<dim, spacedim, LAC> &interface,
                                                         const MPI_Comm &comm) :
  dealii::ParameterAcceptor(name),
  comm(comm),
  interface(interface),
  pde(name,interface,MPI_COMM_WORLD),
  eh("Error handler",interface.get_component_names(),
     print(std::vector<std::string>(interface.n_components,"L2,H1,Linfty"),";")),
  exact_solution("Exact solution", interface.n_components),
  solver("Solver")
{
  dealii::ParameterAcceptor::add_parameter("Number of cycles", n_cycles);
  dealii::ParameterAcceptor::add_parameter("Starting time", start_time);
  dealii::ParameterAcceptor::add_parameter("Final time", final_time);
  dealii::ParameterAcceptor::add_parameter("Time step", time_step);
}

template<int dim, int spacedim, typename LAC>
void LinearQuasiStaticProblem<dim,spacedim,LAC>::init()
{
  pde.init();
}

template<int dim, int spacedim, typename LAC>
void LinearQuasiStaticProblem<dim,spacedim,LAC>::run()
{
  init();
  std::vector<double> jacobian_coefficients(interface.n_components, 0);
  Assert(jacobian_coefficients.size(),
         ExcInternalError("Expecting at least one solution vector in the interface."));
  jacobian_coefficients[0] = 1.0;
  std::string sol_name = interface.solution_names[0];

  // Matrix name
  std::string mat_name = interface.matrices_names[0];


  auto &solution = pde.v(sol_name);

  solution = 1.0;

  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    {
      unsigned int time_step_number=0;
      pde.update_functions_and_constraints(start_time);
      for (double t=start_time; t<=final_time; t+= time_step, ++time_step_number)
        {
          auto residual(pde.v(sol_name));
          auto update = pde.v(sol_name);

          pde.constraints[0]->distribute(solution);
          interface.set_jacobian_coefficients(jacobian_coefficients);

          pde.assemble_matrices();
          pde.residual(residual);
          residual *= -1;

          const auto &A = pde.m(mat_name);
          update = 0;
          if (pde.use_direct_solver)
            {
              if (typeid(LAC) == typeid(LADealII))
                {
                  pde.pcout << "Solving using direct solver and LADealII" << std::endl;
                  SparseDirectUMFPACK inv;
                  inv.initialize((LADealII::BlockMatrix &)A);
                  inv.vmult((LADealII::VectorType &)update,
                            (LADealII::VectorType &)residual);
                }
              else if (typeid(LAC) == typeid(LATrilinos)
                       && interface.n_components == 1)
                {
                  pde.pcout << "Solving using Trilinos direct solver" << std::endl;
                  TrilinosWrappers::SolverDirect inv(solver.control);
                  inv.solve((TrilinosWrappers::SparseMatrix &)A.block(0,0),
                            (TrilinosWrappers::MPI::Vector &) update.block(0),
                            (TrilinosWrappers::MPI::Vector &) residual.block(0));
                }
              else
                AssertThrow(false, ExcNotImplemented());
            }
          else
            {
              pde.pcout << "Solving using iterative solver" << std::endl;
              solver.op = linear_operator<typename LAC::VectorType>(A);
              solver.prec = identity_operator<typename LAC::VectorType>(solver.op);

              solver.parse_parameters_call_back();
              solver.vmult(update, residual);
            }

          solution += update;
          pde.constraints[0]->distribute(solution);

          std::stringstream s;
          if (n_cycles > 1)
            s << "_" << cycle;

          if (time_step > 0)
            s <<  "." << time_step_number;

          pde.output_solution(s.str());

          // Make sure we exit if the time_step is zero
          if (time_step == 0)
            break;
        }
      eh.error_from_exact(interface.get_error_mapping(),
                          interface.get_dof_handler(),
                          solution,
                          exact_solution);
      eh.output_table(pde.pcout);

      // Now refine the grid, only if we are not on last trip.
      if (cycle < n_cycles-1)
        {
          pde.update_functions_and_constraints(start_time);
          pde.refine_mesh();
          pde.setup_dofs(false);
        }
    }
  eh.output_table(pde.pcout);
}


#define INSTANTIATE(dim,spacedim,LA) \
  template class LinearQuasiStaticProblem<dim,spacedim,LA>;

PIDOMUS_INSTANTIATE(INSTANTIATE)

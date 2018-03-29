#include "linear_imex_problem.h"
#include "pidomus_macros.h"

#include <deal.II/lac/sparse_direct.h>

#include <deal2lkit/utilities.h>

template<int dim, int spacedim, typename LAC>
LinearIMEXProblem<dim,spacedim,LAC>::LinearIMEXProblem(const std::string &name,
                                                       PDEBaseInterface<dim, spacedim, LAC> &interface,
                                                       const MPI_Comm &comm) :
  dealii::ParameterAcceptor(name),
  comm(comm),
  interface(interface),
    pde(name,interface,MPI_COMM_WORLD),
    eh("Error handler",interface.get_component_names(),
       print(std::vector<std::string>(interface.n_components,"L2,H1,Linfty"),";")),
    exact_solution("Problem data -- Exact solution", interface.n_components),
    initial_solution("Problem data -- Initial solution", interface.n_components)
{
  Assert(interface.n_vectors == 2,
         ExcMessage("This solver only works if your interface uses two vectors"));

  dealii::ParameterAcceptor::add_parameter("Number of cycles", n_cycles);
  dealii::ParameterAcceptor::add_parameter("Starting time", start_time);
  dealii::ParameterAcceptor::add_parameter("Final time", final_time);
  dealii::ParameterAcceptor::add_parameter("Time step", time_step);
}

template<int dim, int spacedim, typename LAC>
void LinearIMEXProblem<dim,spacedim,LAC>::init()
{
  pde.init();
}

template<int dim, int spacedim, typename LAC>
void LinearIMEXProblem<dim,spacedim,LAC>::run()
{
  init();
  std::vector<double> jacobian_coefficients(interface.n_vectors, 0);
  jacobian_coefficients[0] = 1.0;

  AssertDimension(interface.n_vectors,2);

  auto &solution = pde.v(0);
  auto &previous_solution = pde.v(1);
  pde.interpolate_or_project(initial_solution, solution);

  for (unsigned int cycle=0; cycle < n_cycles; ++cycle)
    {
      unsigned int time_step_number=0;
      pde.update_functions_and_constraints(start_time,time_step);
      for (double t=start_time; t<=final_time; t+= time_step, ++time_step_number)
        {
          previous_solution = solution;

          if (t > start_time)
            {
              pde.update_functions_and_constraints(t,time_step);

              auto residual(pde.v(0));
              auto update = pde.v(0);

              pde.C(0).distribute(solution);
              interface.set_jacobian_coefficients(jacobian_coefficients);

              if (time_step_number == 1)
                {
                  pde.assemble_matrices();
                  pde.initialize_solver();
                }
              pde.residual(residual);
              residual *= -1;

              pde.solve(update, residual);
              solution += update;

              pde.C(0).distribute(solution);
            }

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
          pde.update_functions_and_constraints(start_time,time_step);
          pde.refine_mesh();
          pde.setup_dofs(false);
        }
    }
}


#define INSTANTIATE(dim,spacedim,LA) \
  template class LinearIMEXProblem<dim,spacedim,LA>;

PIDOMUS_INSTANTIATE(INSTANTIATE)

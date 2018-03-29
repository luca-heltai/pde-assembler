#include "pde_handler.h"
#include "tests.h"

#include "pde_system_interface.h"
#include "linear_imex_problem.h"

#include <deal.II/lac/sparse_direct.h>


template <int dim, int spacedim, typename LAC=LADealII>
class TimeDependentStokes : public PDESystemInterface<dim,spacedim, TimeDependentStokes<dim,spacedim,LAC>, LAC>
{

public:
  TimeDependentStokes () :
    PDESystemInterface<dim,spacedim,TimeDependentStokes<dim,spacedim,LAC>, LAC >
    ("Stokes problem", (dim == 2 ? std::vector<std::string>
  {"u","u","p"
  } :
  std::vector<std::string> {"u","u","u","p"}),
  {"system"}, {"solution","previous_solution"},"FESystem[FE_Q(2)^d-FE_DGP(1)]"),
  velocity(0),
           pressure(dim)
  {}

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &fe_cache,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &residuals,
                              bool compute_only_system_terms) const
  {
    ResidualType et = 0;
    this->reinit (et, cell, fe_cache);

    const double dt = this->get_current_time_step();

    auto &epsu  = fe_cache.get_symmetric_gradients("solution", velocity, et);
    auto &p     = fe_cache.get_values("solution", pressure, et);
    auto &u_pre = fe_cache.get_values("previous_solution", velocity, double(0));
    auto &u     = fe_cache.get_values("solution", velocity, et);

    const unsigned int n_q_points = epsu.size();
    const auto &JxW = fe_cache.get_JxW_values();
    const auto &fev = fe_cache.get_current_fe_values();

    for (unsigned int i=0; i<fev.dofs_per_cell; ++i)
      {
        residuals[0][i] = 0;
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            const auto divu = trace(epsu[q]);
            const auto v = fev[velocity].value(i,q);
            const auto dp = fev[pressure].value(i,q);
            const auto divv = fev[velocity].divergence(i,q);
            const auto epsv = fev[velocity].symmetric_gradient(i,q);

            residuals[0][i] += (
                                 (u[q]-u_pre[q])*v/dt +
                                 scalar_product(epsu[q],epsv)-divv*p[q]
                                 -divu*dp
                               )*JxW[q];
          }
      }
  }

  const FEValuesExtractors::Vector velocity;
  const FEValuesExtractors::Scalar pressure;
};


using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog(true);
  deallog.depth_file(1);
  deallog.depth_console(1);

  const int dim = 2;
  const int spacedim = 2;

  TimeDependentStokes<dim,spacedim,LADealII> stokes;
  LinearIMEXProblem<dim,spacedim,LADealII> problem("/IMEX/", stokes);

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_handler_05.prm",
                                           // SOURCE_DIR "/parameters/pde_handler_05.prm");
                                           "used_parameters.prm");

  problem.run();
}

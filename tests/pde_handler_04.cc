#include "pde_handler.h"
#include "tests.h"

#include "pde_system_interface.h"
#include "quasi_static_problem.h"

#include <deal.II/lac/sparse_direct.h>


template <int dim, int spacedim, typename LAC=LADealII>
class NavierStokes : public PDESystemInterface<dim,spacedim, NavierStokes<dim,spacedim,LAC>, LAC>
{

public:
  NavierStokes () :
    PDESystemInterface<dim,spacedim,NavierStokes<dim,spacedim,LAC>, LAC >
    ("Stokes problem", (dim == 2 ? std::vector<std::string> {"u","u","p"} :
  std::vector<std::string> {"u","u","u","p"}),
  {"system"}, {"solution"},"FESystem[FE_Q(2)^d-FE_DGP(1)]")
  {};

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &fe_cache,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &residuals,
                              bool compute_only_system_terms) const
  {
    const FEValuesExtractors::Vector velocity(0);
    const FEValuesExtractors::Scalar pressure(dim);

    EnergyType et = 0;
    ResidualType rt = 0;
    this->reinit (et, cell, fe_cache);
    this->reinit (rt, cell, fe_cache);
    auto &epsu  = fe_cache.get_symmetric_gradients("solution", velocity, et);
    auto &p     = fe_cache.get_values("solution", pressure, et);

    auto &gradu  = fe_cache.get_gradients("solution", velocity, rt);
    auto &u = fe_cache.get_values("solution", velocity, rt);

    const unsigned int n_q_points = epsu.size();
    const auto &JxW = fe_cache.get_JxW_values();
    const auto &fev = fe_cache.get_current_fe_values();

    energies[0] = 0;
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        auto divu = trace(epsu[q]);
        energies[0] += (.5*scalar_product(epsu[q],epsu[q])-divu*p[q])*JxW[q];
        for (unsigned int j=0; j<fev.dofs_per_cell; ++j)
          {
            const auto &v = fev[velocity].value(j,q);
            residuals[0][j] += (gradu[q]*u[q])*v*JxW[q];
          }
      }
  }
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

  NavierStokes<dim,spacedim,LADealII> navier_stokes;
  QuasiStaticProblem<dim,spacedim,LADealII> problem("/Quasi static/", navier_stokes);

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_handler_04.prm",
                                           //SOURCE_DIR "/parameters/pde_handler_02.prm");
                                           "used_parameters.prm");

  problem.run();
}

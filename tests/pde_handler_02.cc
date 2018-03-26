#include "pde_handler.h"
#include "tests.h"

#include "pde_system_interface.h"
#include "linear_quasi_static_problem.h"

#include <deal.II/lac/sparse_direct.h>


template <int dim, int spacedim, typename LAC=LADealII>
class Poisson : public PDESystemInterface<dim,spacedim, Poisson<dim,spacedim,LAC>, LAC>
{

public:
  Poisson () :
    PDESystemInterface<dim,spacedim,Poisson<dim,spacedim,LAC>, LAC >
    ("Poisson problem", {"u"}, {"system"}, {"solution"},"FESystem[FE_Q(1)]")
  {};

  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &fe_cache,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &residuals,
                              bool compute_only_system_terms) const
  {
    const FEValuesExtractors::Scalar scalar(0);

    EnergyType et = 0;
    this->reinit (et, cell, fe_cache);
    auto &gradu = fe_cache.get_gradients("solution", scalar, et);

    const unsigned int n_q_points = gradu.size();
    const auto &JxW = fe_cache.get_JxW_values();

    energies[0] = 0;
    for (unsigned int q=0; q<n_q_points; ++q)
      energies[0] += .5*gradu[q]*gradu[q]*JxW[q];
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

  Poisson<dim,spacedim,LADealII> poisson;
  LinearQuasiStaticProblem<dim,spacedim,LADealII> problem("/Quasi static/", poisson);

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_handler_02.prm",
                                           //SOURCE_DIR "/parameters/pde_handler_02.prm");
                                           "used_parameters.prm");

  problem.run();
}

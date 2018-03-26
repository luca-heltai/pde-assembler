#include "pde_handler.h"
#include "tests.h"

#include "pde_system_interface.h"
#include "linear_quasi_static_problem.h"

#include <deal.II/lac/sparse_direct.h>


template <int dim, int spacedim, typename LAC=LADealII>
class Stokes : public PDESystemInterface<dim,spacedim, Stokes<dim,spacedim,LAC>, LAC>
{

public:
  Stokes () :
    PDESystemInterface<dim,spacedim,Stokes<dim,spacedim,LAC>, LAC >
    ("Stokes problem", (dim == 2 ? std::vector<std::string>{"u","u","p"} :
                                   std::vector<std::string>{"u","u","u","p"}),
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
    this->reinit (et, cell, fe_cache);
    auto &epsu  = fe_cache.get_symmetric_gradients("solution", velocity, et);
    auto &p     = fe_cache.get_values("solution", pressure, et);

    const unsigned int n_q_points = epsu.size();
    const auto &JxW = fe_cache.get_JxW_values();

    energies[0] = 0;
    for (unsigned int q=0; q<n_q_points; ++q) {
        auto divu = trace(epsu[q]);
        energies[0] += (.5*scalar_product(epsu[q],epsu[q])-divu*p[q])*JxW[q];
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

  Stokes<dim,spacedim,LADealII> stokes;
  LinearQuasiStaticProblem<dim,spacedim,LADealII> problem("/Quasi static/", stokes);

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_handler_03.prm",
                                           //SOURCE_DIR "/parameters/pde_handler_02.prm");
                                           "used_parameters.prm");

  problem.run();
}

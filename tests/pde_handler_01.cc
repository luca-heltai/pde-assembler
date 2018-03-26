#include "pde_handler.h"
#include "tests.h"

#include "pde_system_interface.h"

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

  const int dim = 2;
  const int spacedim = 2;

  Poisson<dim,spacedim,LADealII> poisson;
  PDEHandler<dim,spacedim,LADealII> assembler ("/PDE Assembler/",poisson);
  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_handler_01.prm",
                                           "used_parameters.prm");

  assembler.init();

  std::vector<double> c {1.0};
  poisson.set_jacobian_coefficients(c);

  auto &u = assembler.v("solution");
  auto &A = assembler.m("system");
  // auto &C = assembler.c("system");

  auto res(u);
  u = 1;
  assembler.assemble_matrices();
  deallog << "L2 norm of res: " << res.l2_norm() << std::endl;

  // A.print_formatted(deallog.get_file_stream());

  SparseDirectUMFPACK solver;
  solver.factorize(A);

  assembler.residual(res);
  res *= -1;
  solver.vmult(u,res);
  assembler.constraints[0]->distribute(u);
  // C.distribute(u);
  assembler.output_solution();

  deallog << "Linfty norm of u: " << u.linfty_norm() << std::endl;
}

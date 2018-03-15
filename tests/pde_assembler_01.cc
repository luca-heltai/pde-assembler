#include "pde_handler.h"
#include "tests.h"

#include "pde_assembler_pde_system_interface.h"


template <int dim, int spacedim, typename LAC=LADealII>
class Poisson : public PDEAssemblerPDESystemInterface<dim,spacedim, Poisson<dim,spacedim,LAC>, LAC>
{

public:
  virtual ~Poisson () {};
  Poisson ();


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;


};

template <int dim, int spacedim, typename LAC>
Poisson<dim,spacedim, LAC>::
Poisson():
  PDEAssemblerPDESystemInterface<dim,spacedim,Poisson<dim,spacedim,LAC>, LAC >
  ("Poisson problem",
   1,std::vector<std::string>({"system"}), std::vector<std::string>({"solution"}),
      "FESystem[FE_Q(1)]",
      "u")
{}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
Poisson<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &energies,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{

  const FEValuesExtractors::Scalar s(0);

  EnergyType rt = 0; // dummy number to define the type of variables
  this->reinit (rt, cell, fe_cache);
  auto &gradus = fe_cache.get_gradients("solution", "u", s, rt);

  const unsigned int n_q_points = gradus.size();
  auto &JxW = fe_cache.get_JxW_values();

  for (unsigned int q=0; q<n_q_points; ++q)
      energies[0] += 1/2*gradus[q]*gradus[q] * JxW[q];
}



using namespace dealii;
int main (int argc, char *argv[])
{

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                      numbers::invalid_unsigned_int);

  initlog();
  deallog.depth_file(1);

  const int dim = 2;
  const int spacedim = 3;

  Poisson<dim,spacedim,LADealII> p;
  PDEHandler<dim,spacedim,LADealII> assembler ("PDE Assembler",p);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_assembler_01.prm",
                                SOURCE_DIR "/parameters/pde_assembler_01.prm");



  return 0;
}

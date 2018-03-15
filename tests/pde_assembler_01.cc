#include "pde_handler.h"
#include "tests.h"

#include "pde_assembler_pde_system_interface.h"

#include <deal.II/lac/sparse_direct.h>


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
  ("Poisson problem",1,std::vector<std::string>({"system"}), std::vector<std::string>({"solution"}),
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

  ResidualType rt = 0; // dummy number to define the type of variables
  EnergyType et = 0;
  this->reinit (rt, cell, fe_cache);
  this->reinit (et, cell, fe_cache);
  auto &u = fe_cache.get_values("solution", "u", s, rt);
  auto &gradu = fe_cache.get_gradients("solution", "gradu", s, et);

  const unsigned int n_q_points = u.size();
  auto &JxW = fe_cache.get_JxW_values();
  auto &fe_v = fe_cache.get_current_fe_values();

  energies[0] = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<fe_v.dofs_per_cell; ++i)
        {
          auto v = fe_v[s].value(i,q);
          local_residuals[0][i] += .5*u[q]*v* JxW[q];
        }
      energies[0] += .5*gradu[q]*gradu[q]*JxW[q];
    }
}



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
  PDEHandler<dim,spacedim,LADealII> assembler ("PDE Assembler",poisson);
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters/pde_assembler_01.prm",
                                "used_parameters.prm");

  assembler.init();

  deallog << "Init passed" << std::endl;


  std::map<std::string, double> p;
  std::vector<double> c({1});
  poisson.set_current_parameters_and_coefficients(p,c);

  auto &u = assembler.v("solution");
  auto &A = assembler.m("system");
  u = 1;
  auto res(u);
  assembler.assemble_matrices(res);
  deallog << "L2 norm of res: " << res.l2_norm() << std::endl;

  SparseDirectUMFPACK solver;
  solver.factorize(A);

  res *= -1;
  solver.vmult(u,res);
  assembler.constraints[0]->distribute(u);
  poisson.output_solution();

  deallog << "Linfty norm of u: " << u.linfty_norm() << std::endl;
}

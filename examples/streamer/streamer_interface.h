#ifndef _pidoums_poisson_h_
#define _pidoums_poisson_h_

#include "pde_system_interface.h"

#include <deal2lkit/sacado_tools.h>


template <int dim, int spacedim, typename LAC=LADealII>
class StreamerModel : public PDESystemInterface<dim,spacedim, StreamerModel<dim,spacedim,LAC>, LAC>
{

public:
  virtual ~StreamerModel () {}
  StreamerModel ();

  // interface with the PDESystemInterface :)


  template <typename EnergyType, typename ResidualType>
  void energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                              FEValuesCache<dim,spacedim> &scratch,
                              std::vector<EnergyType> &energies,
                              std::vector<std::vector<ResidualType> > &local_residuals,
                              bool compute_only_system_terms) const;

};

template <int dim, int spacedim, typename LAC>
StreamerModel<dim,spacedim, LAC>::
StreamerModel():
  PDESystemInterface<dim,spacedim,StreamerModel<dim,spacedim,LAC>, LAC >("Streamer model",
      3,1,
      "FESystem[FE_Q(1)^3]",
      "u,u,u","1")
{}



template <int dim, int spacedim, typename LAC>
template <typename EnergyType, typename ResidualType>
void
StreamerModel<dim,spacedim,LAC>::
energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                       FEValuesCache<dim,spacedim> &fe_cache,
                       std::vector<EnergyType> &,
                       std::vector<std::vector<ResidualType> > &local_residuals,
                       bool compute_only_system_terms) const
{

  const FEValuesExtractors::Vector displ(0);

  ResidualType rt = 0; // dummy number to define the type of variables
  this->reinit (rt, cell, fe_cache);
  const double alpha = this->get_alpha();
  fe_cache.cache_local_solution_vector("previous_explicit_solution", this->get_locally_relevant_previous_explicit_solution(),alpha);
  auto &uts = fe_cache.get_values("solution_dot", "u", displ, rt);
  auto &graduts = fe_cache.get_gradients("solution_dot", "du", displ, rt);
  auto &gradus = fe_cache.get_gradients("solution", "du", displ, rt);
  /* auto &us = fe_cache.get_values("solution", "u", displ, rt); */
  /* auto &us_2 = fe_cache.get_values("previous_explicit_solution", "u", displ, alpha); */
  /* auto &us_1 = fe_cache.get_values("explicit_solution", "u", displ, alpha); */

  const unsigned int n_q_points = uts.size();
  auto &JxW = fe_cache.get_JxW_values();

  auto &fev = fe_cache.get_current_fe_values();

//  const double vv = 1e-2;
//  const Tensor<1,spacedim> vv;
//  vv[0] = 1e-2;
//  vv[1] = 0;
//  vv[2] = 0;
  for (unsigned int q=0; q<n_q_points; ++q)
    {
    auto &ut = uts[q];
    /* auto &u_2 = us_2[q]; */
    /* auto &u_1 = us_1[q]; */
      /* auto &u = us[q]; */
      auto &gradu = gradus[q];
      for (unsigned int i=0; i<local_residuals[0].size(); ++i)
        {
	  const double k = 1.;
          auto v = fev[displ].value(i,q);
          auto gradv = fev[displ].gradient(i,q);
          local_residuals[0][i] += (
                                  /* (u*v)*alpha*alpha*1000. -1000.*2.*alpha*alpha*(u_1*v) + (u_2*v)*alpha*alpha*1000. */
                                     /* + */
                                     k*scalar_product(gradu,gradv)
//                + 10.*(ut*v)
                + scalar_product(graduts[q],gradv)
                +ut*v
//                                    + 0.9*(ut[0]-vv)
//                                   -vv
                                   )*JxW[q];
        }

      (void)compute_only_system_terms;

    }

}


#endif

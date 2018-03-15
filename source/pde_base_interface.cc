#include "pde_base_interface.h"
#include "lac/lac_type.h"
#include "copy_data.h"

#include "pidomus_macros.h"

#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
//#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal2lkit/dof_utilities.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/any_data.h>
#include <deal2lkit/utilities.h>
// #include <deal2lkit/sacado_tools.h>
#include <deal2lkit/fe_values_cache.h>

using namespace dealii;
using namespace deal2lkit;
// using namespace SacadoTools;


template <int dim, int spacedim, typename LAC>
PDEBaseInterface<dim,spacedim,LAC>::
PDEBaseInterface(const std::string &name,
                 const unsigned int &ncomp,
                 const std::vector<std::string> &matrices_names,
                 const std::vector<std::string> &solution_names,
                 const std::string &default_fe,
                 const std::string &default_component_names) :
  ParameterAcceptor(name),
  n_components(ncomp),
  n_matrices(matrices_names.size()),
  n_vectors(solution_names.size()),
  matrices_names(matrices_names),
  solution_names(solution_names),
  pfe(name,default_fe,default_component_names,n_components),
  data_out("Output Parameters", "none")
{
  for (unsigned int  i=0; i<solution_names.size(); ++i)
    solution_index[solution_names[i]] = i;
  for (unsigned int  i=0; i<matrices_names.size(); ++i)
    matrix_index[matrices_names[i]] = i;

}

template <int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
init()
{
  matrix_couplings = std::vector<Table<2,DoFTools::Coupling> >(n_matrices);
  build_couplings();
}

template <int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
build_couplings()
{
  std::vector<std::string> str_couplings(n_matrices);
  set_matrix_couplings(str_couplings);

  for (unsigned int i=0; i<n_matrices; ++i)
    {
      std::vector<std::vector<unsigned int> > int_couplings;

      convert_string_to_int(str_couplings[i], int_couplings);

      matrix_couplings[i] = to_coupling(int_couplings);
    }
}


template <int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
convert_string_to_int(const std::string &str_coupling,
                      std::vector<std::vector<unsigned int> > &int_coupling) const
{
  std::vector<std::string> rows = Utilities::split_string_list(str_coupling, ';');
  for (unsigned int r=0; r<rows.size(); ++r)
    {
      std::vector<std::string> str_comp = Utilities::split_string_list(rows[r], ',');
      std::vector<unsigned int> int_comp(str_comp.size());
      for (unsigned int i=0; i<str_comp.size(); ++i)
        int_comp[i] = Utilities::string_to_int(str_comp[i]);

      int_coupling.push_back(int_comp);
    }
}

template <int dim, int spacedim, typename LAC>
Table<2,DoFTools::Coupling>
PDEBaseInterface<dim,spacedim,LAC>::
to_coupling(const std::vector<std::vector<unsigned int> > &coupling_table) const
{
  const unsigned int nc = n_components;
  const unsigned int nb = pfe.n_blocks();
  const std::vector<unsigned int> component_blocks = pfe.get_component_blocks();

  Table<2,DoFTools::Coupling> out_coupling(nc, nc);

  std::vector<DoFTools::Coupling> m(3);
  m[0] = DoFTools::none;
  m[1] = DoFTools::always;
  m[2] = DoFTools::nonzero;

  if (coupling_table.size() == nc)
    for (unsigned int i=0; i<nc; ++i)
      {
        AssertThrow(coupling_table[i].size() == nc, ExcDimensionMismatch(coupling_table[i].size(), nc));
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[coupling_table[i][j]];
      }
  else if (coupling_table.size() == nb)
    for (unsigned int i=0; i<nc; ++i)
      {
        AssertThrow(coupling_table[component_blocks[i]].size() == nb,
                    ExcDimensionMismatch(coupling_table[component_blocks[i]].size(), nb));
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[coupling_table[component_blocks[i]][component_blocks[j]]];
      }
  else if (coupling_table.size() == 0)
    for (unsigned int i=0; i<nc; ++i)
      {
        for (unsigned int j=0; j<nc; ++j)
          out_coupling[i][j] = m[1];
      }
  else
    AssertThrow(false, ExcMessage("You tried to construct a coupling with the wrong number of elements."));

  return out_coupling;
}


template <int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
set_matrix_couplings(std::vector<std::string> &str_couplings) const
{
  std::string ones = print(std::vector<std::string>(pfe.n_blocks(),"1"));

  for (unsigned int m=0; m<n_matrices; ++m)
    {
      for (unsigned int b=0; b<pfe.n_blocks()-1; ++b)
        {
          str_couplings[m] += ones;
          str_couplings[m] += ";";
        }
      str_couplings[m] += ones;
    }
}


template <int dim, int spacedim, typename LAC>
const Table<2,DoFTools::Coupling> &
PDEBaseInterface<dim,spacedim,LAC>::
get_matrix_coupling(const unsigned int &i) const
{
  return matrix_couplings[i];
}



template <int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                FEValuesCache<dim,spacedim> &,
                                std::vector<SSdouble> &,
                                std::vector<std::vector<Sdouble> > &,
                                bool) const

{
  Assert(false, ExcPureFunctionCalled ());
}


template <int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
assemble_energies_and_residuals(const typename DoFHandler<dim,spacedim>::active_cell_iterator &,
                                FEValuesCache<dim,spacedim> &,
                                std::vector<Sdouble> &,
                                std::vector<std::vector<double> > &,
                                bool) const

{
  Assert(false, ExcPureFunctionCalled ());
}



template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
assemble_local_matrices (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         CopyData &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  std::vector<SSdouble> energies(n_matrices);
  std::vector<std::vector<Sdouble> > residuals(n_matrices,
                                               std::vector<Sdouble>(dofs_per_cell));
  assemble_energies_and_residuals(cell,
                                  scratch,
                                  energies,
                                  residuals,
                                  false);

  for (unsigned n=0; n<n_matrices; ++n)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        data.local_residual[i] = 0;
        for (unsigned int k=0; k < n_active_vectors; ++k)
          residuals[n][i] += energies[n].dx(i+dofs_per_cell*k);
      }

  for (unsigned n=0; n<n_matrices; ++n)
    {
      data.local_matrices[n] = 0;
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          data.local_residual[i] = residuals[n][i].val();
          for (unsigned int k=0; k < n_active_vectors; ++k)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              data.local_matrices[n](i,j) +=
                active_coefficients[k]*residuals[n][i].dx(j+dofs_per_cell*k);
        }
    }
}

template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
assemble_local_system_residual (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                                FEValuesCache<dim,spacedim> &scratch,
                                CopyData &data) const
{
  const unsigned dofs_per_cell = data.local_dof_indices.size();

  cell->get_dof_indices (data.local_dof_indices);

  std::vector<Sdouble> energies(n_matrices);
  std::vector<std::vector<double> > residuals(n_matrices,
                                              std::vector<double>(dofs_per_cell));
  assemble_energies_and_residuals(cell,
                                  scratch,
                                  energies,
                                  residuals,
                                  true);

  for (unsigned int k=0; k<n_active_vectors; ++k)
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      residuals[0][i] += energies[0].dx(i+dofs_per_cell*k);

  data.local_residual = residuals[0];
}



template <int dim, int spacedim, typename LAC>
std::string
PDEBaseInterface<dim,spacedim,LAC>::get_component_names() const
{
  return pfe.get_component_names();
}


template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_default_mapping() const
{
  return StaticMappingQ1<dim,spacedim>::mapping;
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_output_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_fe_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_bc_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_kelly_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_error_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_interpolate_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
const Mapping<dim,spacedim> &
PDEBaseInterface<dim,spacedim,LAC>::get_project_mapping() const
{
  return get_default_mapping();
}

template <int dim, int spacedim, typename LAC>
UpdateFlags
PDEBaseInterface<dim,spacedim,LAC>::get_face_update_flags() const
{
  return (update_values         | update_quadrature_points  |
          update_normal_vectors | update_JxW_values);
}

template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::declare_parameters(ParameterHandler &prm)
{
}


template<int dim, int spacedim, typename LAC>
UpdateFlags
PDEBaseInterface<dim,spacedim,LAC>::
get_cell_update_flags() const
{
  return (update_quadrature_points |
          update_JxW_values |
          update_values |
          update_gradients);
}

template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
estimate_error_per_cell(Vector<float> &estimated_error) const
{
//  const DoFHandler<dim,spacedim> &dof = this->get_dof_handler();
//  KellyErrorEstimator<dim,spacedim>::estimate (get_kelly_mapping(),
//                                               dof,
//                                               QGauss <dim-1> (dof.get_fe().degree + 1),
//                                               typename FunctionMap<spacedim>::type(),
//                                               this->get_locally_relevant_solution(),
//                                               estimated_error,
//                                               ComponentMask(),
//                                               0,
//                                               0,
//                                               dof.get_triangulation().locally_owned_subdomain());
}

template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
solution_preprocessing (FEValuesCache<dim,spacedim> & /*scratch*/) const
{}

template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::
output_solution (const std::string &suffix) const
{
  data_out.prepare_data_output( this->get_dof_handler(),
                                suffix);
  auto cnames = Utilities::split_string_list(get_component_names());
  auto &vs = this->get_locally_relevant_solutions();
  auto &us = this->get_solutions();
  for (unsigned int i=0; i<us.size(); ++i)
    *vs[i] = *us[i];

  for (unsigned int i=0; i<n_vectors; ++i)
    {
      std::vector<std::string> names(cnames.size(), solution_names[i]);
      for (unsigned int j=0; j<cnames.size(); ++j)
        names[j] += "_" + cnames[j];
      data_out.add_data_vector (*vs[i], print(names));
    }

  data_out.write_data_and_clear(get_output_mapping());
}

template<int dim, int spacedim, typename LAC>
void PDEBaseInterface<dim,spacedim,LAC>::set_current_parameters_and_coefficients
(const std::map<std::string, double> &parameters,
 const std::vector<double> &coefficients) const
{
  AssertDimension(n_vectors, coefficients.size());
  current_coefficients = coefficients;
  current_parameters = parameters;
  const auto &v = this->get_solutions();

  active_vectors.resize(0);
  passive_vectors.resize(0);

  active_vector_names.resize(0);
  passive_vector_names.resize(0);

  active_coefficients.resize(0);
  for (unsigned int i=0; i<n_vectors; ++i)
    {
      if (current_coefficients[i] == 0)
        {
          passive_vectors.push_back(v[i].get());
          passive_vector_names.push_back(solution_names[i]);
        }
      else
        {
          active_vectors.push_back(v[i].get());
          active_vector_names.push_back(solution_names[i]);
          active_coefficients.push_back(current_coefficients[i]);
        }
    }
  n_active_vectors = active_vectors.size();
  n_passive_vectors = passive_vectors.size();

  // Sanity checks
  AssertDimension(n_active_vectors+n_passive_vectors, n_vectors);
}

template<int dim, int spacedim, typename LAC>
void
PDEBaseInterface<dim,spacedim,LAC>::connect_to_signals() const
{}


#define INSTANTIATE(dim,spacedim,LAC) \
  template class PDEBaseInterface<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)



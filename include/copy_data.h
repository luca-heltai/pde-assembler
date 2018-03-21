#ifndef pidomus_copy_data_h
#define pidomus_copy_data_h

#include <deal.II/lac/full_matrix.h>

using namespace dealii;

namespace pidomus
{
  struct CopyData
  {
    CopyData (const unsigned int &dofs_per_cell,
              const unsigned int &n_matrices)
      :
      local_dof_indices  (dofs_per_cell),
      local_residual     (dofs_per_cell),
      local_matrices     (n_matrices,
                          FullMatrix<double>(dofs_per_cell,
                                             dofs_per_cell))
    {}

    CopyData (const CopyData &data)
      :
      local_dof_indices  (data.local_dof_indices),
      local_residual     (data.local_residual),
      local_matrices     (data.local_matrices)
    {}

    ~CopyData()
    {}

    std::vector<types::global_dof_index>  local_dof_indices;
    std::vector<double>                   local_residual;
    std::vector<FullMatrix<double> >      local_matrices;
  };
}

#endif

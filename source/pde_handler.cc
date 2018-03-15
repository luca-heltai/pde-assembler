#include "pde_handler.h"

#include "pidomus_macros.h"
#include "pidomus_signals.h"

#include <deal.II/base/logstream.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/block_linear_operator.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>


#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal2lkit/utilities.h>
#include "lac/lac_initializer.h"

#include <typeinfo>
#include <limits>
#include <numeric>

#include "tria_helper.h"


using namespace dealii;
using namespace deal2lkit;


// This file contains the implementation of:
// - constructor
// - run()
// - make_grid_fe()
// - setup_dofs()
// - solve_jacobian_system()
//

template <int dim, int spacedim, typename LAC>
PDEHandler<dim, spacedim, LAC>::PDEHandler (const std::string &name,
                                            const PDEBaseInterface<dim, spacedim, LAC> &interface,
                                            const MPI_Comm &communicator)
  :
  ParameterAcceptor(name),
  comm(communicator),
  interface(interface),
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(comm)
          == 0)),

  tria_helper(comm),

  pgr("Refinement"),

  constraints(interface.n_matrices),

  n_matrices(interface.n_matrices),

  forcing_terms("Forcing terms",
                interface.n_components,
                interface.get_component_names(), ""),
  neumann_bcs("Neumann boundary conditions",
              interface.n_components,
              interface.get_component_names(), ""),
  dirichlet_bcs("Dirichlet boundary conditions",
                interface.n_components,
                interface.get_component_names(), "0=ALL"),
  dirichlet_bcs_dot("Time derivative of Dirichlet boundary conditions",
                    interface.n_components,
                    interface.get_component_names(), ""),

  zero_average("Zero average constraints",
               interface.n_components,
               interface.get_component_names() ),


  we_are_parallel(Utilities::MPI::n_mpi_processes(comm) > 1)
{

  interface.initialize_simulator (*this);

  constraints[0] = SP(new ConstraintMatrix());

  for (unsigned int i=1; i<n_matrices; ++i)
    constraints[i] = constraints[0];

  for (unsigned int i=0; i<n_matrices; ++i)
    {
      matrices.push_back( SP( new typename LAC::BlockMatrix() ) );
      matrix_sparsities.push_back( SP( new typename LAC::BlockSparsityPattern() ) );
    }
}

template<int dim, int spacedim, typename LAC>
void PDEHandler<dim,spacedim,LAC>::init()
{
  interface.connect_to_signals();
  make_grid_fe();
  setup_dofs(true);
  constraints[0]->distribute(*solutions[0]);
}

template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::make_grid_fe()
{
  auto _timer = computing_timer.scoped_timer("Make grid and finite element");
  signals.begin_make_grid_fe();
  tria_helper.make_grid();
  triangulation = tria_helper.get_tria();
  dof_handler = SP(new DoFHandler<dim, spacedim>(*triangulation));
  signals.postprocess_newly_created_triangulation(triangulation.get());
  fe = interface.pfe();
  triangulation->refine_global (initial_global_refinement);
  signals.end_make_grid_fe();
}



template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::setup_dofs (const bool &first_run)
{
  auto _timer = computing_timer.scoped_timer("Setup dof systems");
  signals.begin_setup_dofs();
  std::vector<unsigned int> sub_blocks = interface.pfe.get_component_blocks();
  dof_handler->distribute_dofs (*fe);
  DoFRenumbering::component_wise (*dof_handler, sub_blocks);

  dofs_per_block.clear();
  dofs_per_block.resize(interface.pfe.n_blocks());

  DoFTools::count_dofs_per_block (*dof_handler, dofs_per_block,
                                  sub_blocks);

  std::locale s = pcout.get_stream().getloc();
  pcout.get_stream().imbue(std::locale(""));
  pcout << "Number of active cells: "
        << triangulation->n_global_active_cells()
        << " (on "
        << triangulation->n_levels()
        << " levels)"
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler->n_dofs()
        << "(" << print(dofs_per_block, "+") << ")"
        << std::endl
        << std::endl;
  pcout.get_stream().imbue(s);


  partitioning.resize(0);
  relevant_partitioning.resize(0);

  IndexSet relevant_set;
  {
    global_partitioning = dof_handler->locally_owned_dofs();
    for (unsigned int i = 0; i < interface.pfe.n_blocks(); ++i)
      partitioning.push_back(global_partitioning.get_view( std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                           std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));

    DoFTools::extract_locally_relevant_dofs (*dof_handler,
                                             relevant_set);

    for (unsigned int i = 0; i < interface.pfe.n_blocks(); ++i)
      relevant_partitioning.push_back(relevant_set.get_view(std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i, 0),
                                                            std::accumulate(dofs_per_block.begin(), dofs_per_block.begin() + i + 1, 0)));
  }

  ScopedLACInitializer initializer(dofs_per_block,
                                   partitioning,
                                   relevant_partitioning,
                                   comm);

  AssertDimension(solutions.size(), locally_relevant_solutions.size());
  for (unsigned int i=0; i<solutions.size(); ++i)
    {
      initializer(*solutions[i]);
      if (we_are_parallel)
        initializer.ghosted(*locally_relevant_solutions[i]);
      else
        initializer(*locally_relevant_solutions[i]);
    }

  for (unsigned int i=0; i < n_matrices; ++i)
    {
      matrices[i]->clear();
      initializer(*matrix_sparsities[i],
                  *dof_handler,
                  *constraints[i],
                  interface.get_matrix_coupling(i));
      matrices[i]->reinit(*matrix_sparsities[i]);
    }

//  if (first_run)
//    {
//      if (fe->has_support_points())
//        {
//          VectorTools::interpolate(interface.get_interpolate_mapping(), *dof_handler, initial_solution, solution);
//          VectorTools::interpolate(interface.get_interpolate_mapping(), *dof_handler, initial_solution_dot, solution_dot);
//        }
//      else if (!we_are_parallel)
//        {
//          const QGauss<dim> quadrature_formula(fe->degree + 1);
//          VectorTools::project(interface.get_project_mapping(), *dof_handler, *constraints[0], quadrature_formula, initial_solution, solution);
//          VectorTools::project(interface.get_project_mapping(), *dof_handler, *constraints[0], quadrature_formula, initial_solution_dot, solution_dot);
//        }
//      else
//        {
//          Point<spacedim> p;
//          Vector<double> vals(interface.n_components);
//          Vector<double> vals_dot(interface.n_components);
//          initial_solution.vector_value(p, vals);
//          initial_solution_dot.vector_value(p, vals_dot);

//          unsigned int comp = 0;
//          for (unsigned int b=0; b<solution.n_blocks(); ++b)
//            {
//              solution.block(b) = vals[comp];
//              solution_dot.block(b) = vals_dot[comp];
//              comp += fe->element_multiplicity(b);
//            }
//        }

//      signals.fix_initial_conditions(solution, solution_dot);
//      locally_relevant_explicit_solution = solution;

//    }
  signals.end_setup_dofs();
}



template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
apply_neumann_bcs (
  const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
  FEValuesCache<dim,spacedim> &scratch,
  std::vector<double> &local_residual) const
{

  double dummy = 0.0;

  for (unsigned int face=0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      unsigned int face_id = cell->face(face)->boundary_id();
      if (cell->face(face)->at_boundary() && neumann_bcs.acts_on_id(face_id))
        {
          interface.reinit(dummy, cell, face, scratch);

          auto &fev = scratch.get_current_fe_values();
          auto &q_points = scratch.get_quadrature_points();
          auto &JxW = scratch.get_JxW_values();

          for (unsigned int q=0; q<q_points.size(); ++q)
            {
              Vector<double> T(interface.n_components);
              neumann_bcs.get_mapped_function(face_id)->vector_value(q_points[q], T);

              for (unsigned int i=0; i<local_residual.size(); ++i)
                for (unsigned int c=0; c<interface.n_components; ++c)
                  local_residual[i] -= T[c]*fev.shape_value_component(i,q,c)*JxW[q];

            }// end loop over quadrature points

          break;

        } // endif face->at_boundary

    }// end loop over faces

}// end function definition



template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
apply_forcing_terms (const typename DoFHandler<dim,spacedim>::active_cell_iterator &cell,
                     FEValuesCache<dim,spacedim> &scratch,
                     std::vector<double> &local_residual) const
{
  unsigned cell_id = cell->material_id();
  if (forcing_terms.acts_on_id(cell_id))
    {
      double dummy = 0.0;
      interface.reinit(dummy, cell, scratch);

      auto &fev = scratch.get_current_fe_values();
      auto &q_points = scratch.get_quadrature_points();
      auto &JxW = scratch.get_JxW_values();
      for (unsigned int q=0; q<q_points.size(); ++q)
        for (unsigned int i=0; i<local_residual.size(); ++i)
          for (unsigned int c=0; c<interface.n_components; ++c)
            {
              double B = forcing_terms.get_mapped_function(cell_id)->value(q_points[q],c);
              local_residual[i] -= B*fev.shape_value_component(i,q,c)*JxW[q];
            }
    }
}


template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim,spacedim,LAC>::
apply_dirichlet_bcs (const DoFHandler<dim,spacedim> &dof_handler,
                     const ParsedDirichletBCs<dim,spacedim> &bc,
                     ConstraintMatrix &constraints) const
{
  try
    {
      bc.interpolate_boundary_values(interface.get_bc_mapping(),dof_handler,constraints);
    }
  catch (...)
    {
      AssertThrow(!we_are_parallel,
                  ExcMessage("You called VectorTools::project_boundary_values(), which is not \n"
                             "currently supported on deal.II in parallel settings.\n"
                             "Feel free to submit a patch :)"));
      const QGauss<dim-1> quad(fe->degree+1);
      bc.project_boundary_values(interface.get_bc_mapping(),dof_handler,quad,constraints);
    }
  unsigned int codim = spacedim - dim;
  // if (codim == 0)
  //   bc.compute_nonzero_normal_flux_constraints(dof_handler,interface.get_bc_mapping(),constraints);
}



template <int dim, int spacedim, typename LAC>
void PDEHandler<dim, spacedim, LAC>::assemble_matrices
(const std::map<std::string, double>         &parameters,
 const std::vector<double>                   &coefficients,
 const std::vector<std::shared_ptr<LADealII::VectorType>>  &input_vectors)
{
  auto _timer = computing_timer.scoped_timer ("Assemble matrices");

  signals.begin_assemble_matrices();

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);

  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;


  for (unsigned int i=0; i<n_matrices; ++i)
    *(matrices[i]) = 0;



  auto local_copy = [this]
                    (const pidomus::CopyData & data)
  {

    for (unsigned int i=0; i<n_matrices; ++i)
      this->constraints[i]->distribute_local_to_global (data.local_matrices[i],
                                                        data.local_dof_indices,
                                                        *(this->matrices[i]));
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_matrices(cell, scratch, data);
  };



  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));

  for (unsigned int i=0; i<n_matrices; ++i)
    matrices[i]->compress(VectorOperation::add);
}



template <int dim, int spacedim, typename LAC>
void
PDEHandler<dim, spacedim, LAC>::residual(const std::vector<double>                                           &parameters,
                                           const std::vector<double>                                           &coefficients,
                                           const std::vector<const std::shared_ptr<typename LAC::VectorType>>  &input_vectors,
                                           typename LAC::VectorType                                            &dst,
                                           bool setup_jacobian)
{
  auto _timer = computing_timer.scoped_timer ("Residual");

  signals.begin_residual();

  // syncronize(t,solution,solution_dot);

  const QGauss<dim> quadrature_formula(fe->degree + 1);
  const QGauss < dim - 1 > face_quadrature_formula(fe->degree + 1);


  FEValuesCache<dim,spacedim> fev_cache(interface.get_fe_mapping(),
                                        *fe, quadrature_formula,
                                        interface.get_cell_update_flags(),
                                        face_quadrature_formula,
                                        interface.get_face_update_flags());

  interface.solution_preprocessing(fev_cache);

  dst = 0;

  auto local_copy = [&dst, this] (const pidomus::CopyData & data)
  {

    this->constraints[0]->distribute_local_to_global (data.local_residual,
                                                      data.local_dof_indices,
                                                      dst);
  };

  auto local_assemble = [ this ]
                        (const typename DoFHandler<dim, spacedim>::active_cell_iterator & cell,
                         FEValuesCache<dim,spacedim> &scratch,
                         pidomus::CopyData & data)
  {
    this->interface.assemble_local_system_residual(cell,scratch,data);
    // apply conservative loads
    this->apply_forcing_terms(cell, scratch, data.local_residual);

    if (cell->at_boundary())
      this->apply_neumann_bcs(cell, scratch, data.local_residual);
  };

  typedef
  FilteredIterator<typename DoFHandler<dim, spacedim>::active_cell_iterator>
  CellFilter;
  WorkStream::
  run (CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->begin_active()),
       CellFilter (IteratorFilters::LocallyOwnedCell(),
                   dof_handler->end()),
       local_assemble,
       local_copy,
       fev_cache,
       pidomus::CopyData(fe->dofs_per_cell,n_matrices));


  dst.compress(VectorOperation::add);

  auto id = solutions[0]->locally_owned_elements();
  for (unsigned int i = 0; i < id.n_elements(); ++i)
    {
      auto j = id.nth_index_in_set(i);
      if (constraints[0]->is_constrained(j))
        dst[j] = (*solutions[0])(j) - (*locally_relevant_solutions[0])(j);
    }

  dst.compress(VectorOperation::insert);

  signals.end_residual();
}




#define INSTANTIATE(dim,spacedim,LAC) \
  template class PDEHandler<dim,spacedim,LAC>;


PIDOMUS_INSTANTIATE(INSTANTIATE)



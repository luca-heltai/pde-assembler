#ifndef __sak_conservative_interface_h
#define __sak_conservative_interface_h

#include "interface.h"

template<int dim, int spacedim, int n_components, class Implementation>
class ConservativeInterface : public Interface<dim,spacedim,n_components>
{

  typedef Assembly::Scratch::NFields<dim,spacedim> Scratch;
  typedef Assembly::CopyData::NFieldsPreconditioner<dim,spacedim> CopyPreconditioner;
  typedef Assembly::CopyData::NFieldsSystem<dim,spacedim> CopySystem;
public:

  ConservativeInterface(const std::string &name="",
                        const std::string &default_fe="FE_Q(1)",
                        const std::string &default_component_names="u",
                        const std::string &default_coupling="",
                        const std::string &default_preconditioner_coupling="") :
    Interface<dim,spacedim,n_components>(name, default_fe, default_component_names,
                                         default_coupling, default_preconditioner_coupling) {};

  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                 Scratch &s,
                                 CopySystem &d,
                                 Sdouble &e) const
  {
    static_cast<const Implementation *>(this)->system_energy(c,s,d,e);
  }


  virtual void get_system_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                 Scratch &s,
                                 CopySystem &d,
                                 SSdouble &e) const
  {
    static_cast<const Implementation *>(this)->system_energy(c,s,d,e);
  }


  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                         Scratch &s,
                                         CopySystem &d,
                                         Sdouble &e) const
  {
    static_cast<const Implementation *>(this)->preconditioner_energy(c,s,d,e);
  }


  virtual void get_preconditioner_energy(const typename DoFHandler<dim,spacedim>::active_cell_iterator &c,
                                         Scratch &s,
                                         CopyPreconditioner &d,
                                         SSdouble &e) const
  {
    static_cast<const Implementation *>(this)->preconditioner_energy(c,s,d,e);
  }
};

#endif

#!/usr/bin/env python3

import sys
import shutil
import os
import glob
from sys import platform
from sympy import *
from sympy.parsing.sympy_parser import parse_expr

def main(args=None):
    """The main routine."""

# To add a test you need to adjust 
#   -testnames
#   -pre_d
#   -pre_v
#   -output

    pre_v = "-1/(2*pi)*sin(2*pi*x)**2*sin(2*pi*y)**2"  
    press = "sin(2*pi*x)*sin(2*pi*y)"

    non_linear = False
    
    #input handling
    if args is None:
        args = sys.argv[1:]

    if len(args) > 0:
        pre_v = args[0]
    if len(args) > 1:
        press = args[1]

    # set up symbolic variables to be derived later
    x, y, z, t = var('x,y,z,t')
    local_dict = {"x": x, "y": y, "z": z, "t": t}
    inv_dict = {"x": y, "y": x, "z": z, "t": t}

    pre_v_ = parse_expr(pre_v, local_dict=local_dict)
    p = parse_expr(press, local_dict=local_dict)
    pt = p.diff(t)

    # curl operator \nabla \times v
    v = [-pre_v_.diff(y), pre_v_.diff(x)]
    vt = [vi.diff(t) for vi in v]

    f_navier = [v[0].diff(t) + (v[0])*v[0].diff(x) +
    (v[1])*v[0].diff(y), v[1].diff(t) + (v[0])*v[1].diff(x) +
    (v[1])*v[1].diff(y), 0]
        
    f_stokes = [-v[0].diff(x, 2) - v[0].diff(y, 2) + p.diff(x),
    -v[1].diff(x, 2) - v[1].diff(y, 2) + p.diff(y), 0]

    total_f = [ f for f in f_stokes ]
    if non_linear:
        for i in range(len(total_f)):
            total_f[i] += f_navier[i]
    
    # specify prm file
    v_str = ("subsection Problem data -- Dirichlet boundary conditions (u,u,p)\n"   
                 +"  set IDs and component masks = 0=u\n"   
                 +"  set IDs and expressions     = 0=" 
                 +  str(v[0]) + "; " + str(v[1]) + "; " + str(p) + "\n" 
                 +"  set Used constants          = \nend\n" +
                 "subsection Problem data -- Exact solution\n"
                 +"  set Function expression = " 
                 + str(v[0]) + "; " + str(v[1]) + "; " + str(p) + "\nend\n" +
                 "subsection Problem data -- Forcing terms (u,u,p)\n"   
                 +"  set IDs and component masks = 0=ALL\n"   
                 +"  set IDs and expressions     = 0=" 
                 + str(total_f[0]) + "; " + str(total_f[1]) + "; "
                 + str(total_f[2]) +"\n"
                 +"  set Used constants          = \nend\n" +
                 "subsection Problem data -- Initial solution\n"   
                 +"  set Function expression ="   
                 + str(v[0].subs(t,0)) + "; " + str(v[1].subs(t,0)) + "; " + str(p.subs(t,0)) + "\nend\n" +  
                 "\nend")
    

    print(v_str.replace("**", "^"))
    
if __name__ == "__main__":
    main()

from operator import truediv

import numpy as np

from Nodes_elements_operations import Nodes_elements_operations
import scipy.sparse as sp


from Finite_element_methods import FEM_Methods
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from Conditions import Boundary_conditions
from scipy.sparse.linalg import splu
import matplotlib
#matplotlib.use('TkAgg')
from scipy.spatial import Delaunay

from Plotting_functions import PlottingFunctions

N_x= 101
N_y=81

L_x=10
L_y=8

Topeni_x=2
Topeni_y=1

lambda_zed = 0.5
lambda_vzduch = 50.0
lambda_topidlo = 1.0
rho_topeni=1800
rho_vzduch=1.3
rho_zed= 1800
c_zed=1000
c_vzduch=1000
c_topeni=1000
alpha_prestup = 10.0  # Součinitel přestupu tepla na vnější stěně
T_okolni = -10.0
T_pocatecni=10
zed=1
Q=600


Nodes_x= np.linspace(0,L_x,N_x)
Nodes_y= np.linspace(0,L_y,N_y)
Nodes_x,Nodes_y = np.meshgrid(Nodes_x,Nodes_y)
Nodes=np.vstack([Nodes_x.ravel(),Nodes_y.ravel()]).T
F_right_side_D= np.zeros(Nodes[:,0].size)
F_right_side_N= np.zeros(Nodes[:,0].size)
tri=Delaunay(Nodes)
elements_idx= tri.simplices

elements= Nodes[elements_idx]
elements_lambda= None

fem_methods = FEM_Methods(Nodes ,elements_idx, L_x, L_y,
      Topeni_x, Topeni_y, lambda_vzduch, lambda_zed, lambda_topidlo, c_topeni,c_zed,c_vzduch, rho_topeni, rho_zed, rho_vzduch)


Global_K_D=fem_methods.Generate_Global_Stiffness_matrix()#fem_methods.GenerateD_matrix()
Global_K_N=fem_methods.Generate_Global_Stiffness_matrix()
Outter_nodes_indicies=Nodes_elements_operations.GenerateOuterNodes(Nodes, L_x, L_y,)
Source_node= Nodes_elements_operations.Generate_point_source_node(L_x, L_y, N_x, N_y,
                                                                Nodes, Topeni_x, Topeni_y,zed)

#Edges_id_horizontal, Edges_id_vertical= Nodes_elements_operations.Generate_edges_couples_for_Newton(Nodes, L_x, L_y, N_x, N_y,)
Boundary_conditions.Give_source(Q, Source_node, F_right_side_D)
Boundary_conditions.Give_source(Q, Source_node, F_right_side_N)
Boundary_conditions.Generate_Dirichlet( Outter_nodes_indicies,Global_K_D,F_right_side_D,T_okolni)
Boundary_conditions.Generate_Newton(Nodes, L_x, L_y, N_x, N_y,F_right_side_N, T_okolni, Global_K_N, alpha_prestup)
T_N= spla.spsolve(Global_K_N, F_right_side_N)
T_D= spla.spsolve(Global_K_D, F_right_side_D)



triangulation = mtri.Triangulation(Nodes[:, 0], Nodes[:, 1], elements_idx)
indisies_profile= np.argwhere(Nodes[:,1]==5)
#PlottingFunctions.Plot_Triagonal(Nodes)
PlottingFunctions.Plot_Thermal_mapping(triangulation, T_D, Nodes, BC='Dirichlet', Element_view=False)
PlottingFunctions.Plot_Thermal_mapping(triangulation, T_N, Nodes, BC='Newton', Element_view=False)
PlottingFunctions.GenerateThermalprofile_stationar(N_x,Nodes, T_D, BC='Dirichlet',indicies=indisies_profile, )
PlottingFunctions.GenerateThermalprofile_stationar(N_x,Nodes, T_N, BC='Newton',indicies=indisies_profile, )
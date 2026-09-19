import numpy as np

from Uloha_2.Nodes_elements_operations import Nodes_elements_operations

from Finite_element_methods import FEM_Methods
import scipy.sparse.linalg as spla
import matplotlib.tri as mtri
from Uloha_2.Conditions import Boundary_conditions
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
l_x_source=0.6
l_y_source= 0.4
elements= Nodes[elements_idx]
elements_lambda= None
velocity=0.3
fem_methods = FEM_Methods(Nodes ,elements_idx, L_x, L_y,
      Topeni_x, Topeni_y, lambda_vzduch, lambda_zed, lambda_topidlo, c_topeni,c_zed,c_vzduch, rho_topeni, rho_zed, rho_vzduch,velocity)


Global_K_D=fem_methods.Generate_Global_Stiffness_matrix()#fem_methods.GenerateD_matrix()
Global_K_N=fem_methods.Generate_Global_Stiffness_matrix_Jet()
Outter_nodes_indicies=Nodes_elements_operations.GenerateOuterNodes(Nodes, L_x, L_y,)


#Edges_id_horizontal, Edges_id_vertical= Nodes_elements_operations.Generate_edges_couples_for_Newton(Nodes, L_x, L_y, N_x, N_y,)
Source_node= Nodes_elements_operations.Generate_point_source_node(L_x, L_y, N_x, N_y,Nodes, Topeni_x, Topeni_y,zed)
(source_couples_vertical,
 source_couples_horizontal) = Nodes_elements_operations.Generate_Source(Nodes, l_x_source, l_y_source,Source_node)
#Boundary_conditions.Generate_Dirichlet( Outter_nodes_indicies,Global_K ,F_right_side,T_out)
Boundary_conditions.Generate_Heat_Source_widespread(Q, F_right_side_N,L_x, L_y, N_x, N_y,source_couples_vertical, source_couples_horizontal,l_x_source, l_y_source)
#Boundary_conditions.Generate_Dirichlet( Outter_nodes_indicies,Global_K_D,F_right_side_D,T_okolni)
Boundary_conditions.Generate_Newton(Nodes, L_x, L_y, N_x, N_y,F_right_side_N, T_okolni, Global_K_N, alpha_prestup)
T_N= spla.spsolve(Global_K_N, F_right_side_N)




triangulation = mtri.Triangulation(Nodes[:, 0], Nodes[:, 1], elements_idx)
indisies_profile= np.argwhere(Nodes[:,1]==5)
#PlottingFunctions.Plot_Triagonal(Nodes)
#PlottingFunctions.Plot_Thermal_mapping(triangulation, T_D, Nodes, BC='Dirichlet', Element_view=False)
PlottingFunctions.Plot_Thermal_mapping(triangulation, T_N, Nodes, BC='Newton', Element_view=False)
#PlottingFunctions.GenerateThermalprofile_stationar(N_x,Nodes, T_D, BC='Dirichlet',indicies=indisies_profile, )
PlottingFunctions.GenerateThermalprofile_stationar(N_x,Nodes, T_N, BC='Newton',indicies=indisies_profile, )
from operator import truediv

import numpy as np

from Nodes_elements_operations import Nodes_elements_operations
import scipy.sparse as sp
from Finite_element_methods import FEM_Methods
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from Time_Iteration_Process import GenerationofTime
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
T_init=10
zed=1
Q=600
Time_steps= 200
t_1 , t_end= 0,100*86400
Time= np.linspace(t_1, t_end, Time_steps)

Nodes_x= np.linspace(0,L_x,N_x)
Nodes_y= np.linspace(0,L_y,N_y)
Nodes_x,Nodes_y = np.meshgrid(Nodes_x,Nodes_y)
Nodes=np.vstack([Nodes_x.ravel(),Nodes_y.ravel()]).T
F_right_side= np.zeros(Nodes[:,0].size)

tri=Delaunay(Nodes)
elements_idx= tri.simplices

elements= Nodes[elements_idx]
elements_lambda= None

fem_methods = FEM_Methods(Nodes ,elements_idx, L_x, L_y,
      Topeni_x, Topeni_y, lambda_vzduch, lambda_zed, lambda_topidlo, c_topeni,c_zed,c_vzduch, rho_topeni, rho_zed, rho_vzduch)


Global_K, Global_M =fem_methods.Generate_Global_Matrices_Time()#fem_methods.GenerateD_matrix()
Outter_nodes_indicies=Nodes_elements_operations.GenerateOuterNodes(Nodes, L_x, L_y,)
Source_node= Nodes_elements_operations.Generate_point_source_node(L_x, L_y, N_x, N_y,
                                                                Nodes, Topeni_x, Topeni_y,zed)

#Edges_id_horizontal, Edges_id_vertical= Nodes_elements_operations.Generate_edges_couples_for_Newton(Nodes, L_x, L_y, N_x, N_y,)
Boundary_conditions.Give_source(Q, Source_node, F_right_side)
#Boundary_conditions.Generate_Dirichlet( Outter_nodes_indicies,Global_K ,F_right_side,T_out)
Boundary_conditions.Generate_Newton(Nodes, L_x, L_y, N_x, N_y,F_right_side, T_okolni, Global_K, alpha_prestup)
T= GenerationofTime.Solve_time_equation(T_init, Global_K ,Global_M, Time, F_right_side)#spla.spsolve(Global_K, F_right_side)

id_points= Nodes_elements_operations.Generate_indicies_of_key_Points(Nodes, L_x, L_y, N_x, N_y)

Thermal_vectors=  Nodes_elements_operations.Generate_Thermal_dependence_in_key_points(id_points, T, Time)



triangulation = mtri.Triangulation(Nodes[:, 0], Nodes[:, 1], elements_idx)

indicies_profile= np.argwhere(Nodes[:,1]==5).reshape(N_x)
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[0], Nodes, Element_view=False, time=Time[0])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[25], Nodes, Element_view=False, time=Time[25])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[50], Nodes, Element_view=False, time=Time[50])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[75], Nodes, Element_view=False, time=Time[75])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[100], Nodes, Element_view=False,time= Time[100])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[125], Nodes, Element_view=False, time=Time[125])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[150], Nodes, Element_view=False, time= Time[150])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[175], Nodes, Element_view=False, time=Time[175])
#PlottingFunctions.Plot_Thermal_mapping_time(triangulation, T[-1], Nodes, Element_view=False, time= Time[-1])


#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[0], indicies_profile, Time[0])
#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[25], indicies_profile, Time[25])
#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[50], indicies_profile, Time[50])
#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[100], indicies_profile, Time[100])
#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[150], indicies_profile, Time[150])
#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[175], indicies_profile, Time[175])
#PlottingFunctions. GenerateThermalProfileUnstationar(N_x, Nodes, T[-1], indicies_profile, Time[-1])

PlottingFunctions.Generate_Time_Thermal_Dependence(Thermal_vectors, id_points, Nodes, Time)
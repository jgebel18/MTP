import numpy as np

import Common_methods
from Common_methods import Common_methods_MTP
import time
import matplotlib



matplotlib.use('TkAgg')







width_nodes= 11
height_nodes=9
Nodes= np.linspace(0,width_nodes*height_nodes-1, width_nodes*height_nodes, dtype=int)
space_step=1
e=600
lambda_t=2
T_out=283.15
final_ratio= (e*space_step**2)/(4*lambda_t)
id_x_source= 24

y
start= time.time()
Iner_Nodes, Outer_Nodes = Common_methods_MTP.Generateindicies(width_nodes, height_nodes)


F = np.zeros(len(Iner_Nodes))
if id_x_source in Iner_Nodes:
    # Najdeme pozici globálního indexu v seznamu vnitřních uzlů
    idx_v_F = np.where(Iner_Nodes == id_x_source)[0][0]
    F[idx_v_F] = final_ratio


Q_matrix, R_matrix = Common_methods_MTP.GenerateNeigbours(Nodes, Iner_Nodes, Outer_Nodes, width_nodes, height_nodes)

Q_0=np.eye(N=Q_matrix[0].size, M=Q_matrix[0].size)
N_Exodus= Common_methods_MTP.IterationProcess(Q_matrix, Q_0)

B= N_Exodus@R_matrix
T_boundary=np.ones(len(Outer_Nodes))*T_out
Full_Grid_1 = Nodes.reshape(height_nodes,width_nodes)
Full_Grid = np.ones((height_nodes, width_nodes)) * T_out
T_from_border=B@T_boundary
T_from_sources=N_Exodus@F
T_field = T_from_sources + T_from_border
T_field_inner=T_field.reshape(height_nodes - 2, width_nodes - 2)
Vector= np.zeros(Nodes.size)
Full_Grid[1:-1, 1:-1] = T_field_inner
end=time.time()
Key_points=np.array([Full_Grid[3][1],Full_Grid[4][5], Full_Grid[7][9],Full_Grid[1][9]])
print(f'Čas výpočtu Metoda Exodus: {end-start} s')
print(f'Teploty v bodech metoda Exodus s teplotou okolí {T_out}[K] :')
for i in range(Key_points.size):
    print(f'Teplota-Bod-{i+1}:{Key_points[i]} [K]')

# T_field má 63 prvků, vnitřní mřížka je (height-2) x (width-2)
Inner_nodes_2= Iner_Nodes.reshape(height_nodes-2,width_nodes-2)
Nodes_2=np.linspace(0,width_nodes-1,width_nodes,dtype=int)
nazev_2=f'MTP-Uloha-7-2-w-{width_nodes}-h-{height_nodes}-idx-{id_x_source}--T_out-{T_out}-1D-curve.png'
nazev=f'MTP-Uloha-7-2-w-{width_nodes}-h-{height_nodes}-idx-{id_x_source}-T_out-{T_out}.png'
Common_methods_MTP.PlotData(nazev, nazev_2, Nodes_2, Full_Grid, f'MTP-Úloha-7-Podúloha-2-Teplotní-mapa-Okolni-Teplota-{T_out}', f'MTP-Úloha-7-Podúloha-2-Teplotní-KřivkaOkolni-Teplota-{T_out}')



import numpy as np
from Nodes_elements_operations import Nodes_elements_operations



class Boundary_conditions:

    @staticmethod
    def Give_source(Q, source_id, F):
        F[source_id]= Q

    @staticmethod
    def Generate_Heat_Source_widespread(Q,F,
                                        L_x, L_y
                                        , N_x, N_y,
                                       edges_id_vertical, edges_id_horizontal, L_x_source,L_y_source ):
        Obvod= 2*L_x_source+2*L_y_source
        l_x = L_x / (N_x - 1)
        l_y = L_y / (N_y - 1)
        q=Q/Obvod
        for i, edge in enumerate(edges_id_horizontal):
            for j, couple in enumerate(edges_id_horizontal[i]):
                idx_a = couple[0]
                idx_b = couple[1]
                F[idx_a] += 1 / 2 * q * l_x
                F[idx_b] += 1 / 2 * q * l_x

        for i, edge in enumerate(edges_id_vertical):
            for j, couple in enumerate(edges_id_vertical[i]):
                idx_a = couple[0]
                idx_b = couple[1]
                F[idx_a] += 1 / 2 * q* l_y
                F[idx_b] += 1 / 2 * q* l_y



    @staticmethod
    def Generate_Dirichlet(key_nodes ,Global_Matrix, F, Temperature  ):
        F[key_nodes]=Temperature
        for index in enumerate(key_nodes):
            Global_Matrix[index, :]=0
            Global_Matrix[index, index] = 1.0

    @staticmethod
    def Generate_Newton( Nodes, L_x, L_y, N_x, N_y, F, T_out, Global_k,alpha):
        edges_id_horizontal, edges_id_vertical = Nodes_elements_operations.Generate_edges_couples_for_Newton(Nodes, L_x,
                                                                                                             L_y, N_x,
                                                                                                             N_y)

        l_x = L_x / (N_x - 1)
        l_y = L_y / (N_y - 1)
        for i, edge in enumerate(edges_id_horizontal):
            for j, couple in enumerate(edges_id_horizontal[i]):
                idx_a=couple[0]
                idx_b=couple[1]
                Global_k[idx_a,idx_a]+=2/6 *alpha*l_x
                Global_k[idx_a, idx_b] += 1 / 6 * alpha * l_x
                Global_k[idx_b, idx_a] += 1 / 6 * alpha * l_x
                Global_k[idx_b, idx_b] += 2 / 6 * alpha * l_x
                F[idx_a]+= 1/2 * alpha * l_x*T_out
                F[idx_b]+= 1/2 * alpha * l_x*T_out
        for i, edge in enumerate(edges_id_vertical):
            for j, couple in enumerate(edges_id_vertical[i]):
                idx_a=couple[0]
                idx_b=couple[1]
                Global_k[idx_a,idx_a]+=2/6 *alpha*l_y
                Global_k[idx_a, idx_b] += 1 / 6 * alpha * l_y
                Global_k[idx_b, idx_a] += 1 / 6 * alpha * l_y
                Global_k[idx_b, idx_b] += 2 / 6 * alpha * l_y
                F[idx_a]+= 1/2 * alpha * l_y*T_out
                F[idx_b]+= 1/2 * alpha * l_y*T_out

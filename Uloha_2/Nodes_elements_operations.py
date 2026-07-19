import numpy as np








class Nodes_elements_operations:


    @staticmethod
    def Generate_indicies_of_key_Points(Nodes,L_x, L_y, N_x, N_y,):
        id_point_1 = np.argwhere(
            (np.isclose(Nodes[:, 0], 1, atol=0.9 * (L_x / (N_x - 1)))) &
            (np.isclose(Nodes[:, 1], 2, atol=0.9 * (L_y / (N_y - 1))))
        )

        id_point_2 = np.argwhere(
            (np.isclose(Nodes[:, 0], 5, atol=0.9 * (L_x / (N_x - 1)))) &
            (np.isclose(Nodes[:, 1], 4, atol=0.9 * (L_y / (N_y - 1))))
        )

        id_point_3 = np.argwhere(
            (np.isclose(Nodes[:, 0], 9, atol=0.9 * (L_x / (N_x - 1)))) &
            (np.isclose(Nodes[:, 1], 7, atol=0.9 * (L_y / (N_y - 1))))
        )

        id_point_4 = np.argwhere(
            (np.isclose(Nodes[:, 0], 9, atol=0.9 * (L_x / (N_x - 1)))) &
            (np.isclose(Nodes[:, 1], 1, atol=0.9 * (L_y / (N_y - 1))))
        )
        return np.array([id_point_1,id_point_2,id_point_3, id_point_4])

    @staticmethod
    def Generate_Thermal_dependence_in_key_points(Key_points, Thermal, Time, ):
        Thermal_vectors= np.zeros((Key_points.size, Time.size))
        for i, idx in enumerate(Key_points):
            for j, t in enumerate(Time):
                Thermal_vectors[i][j]= Thermal[j][idx]
        return  Thermal_vectors



    @staticmethod
    def GenerateOuterNodes(Nodes, Width, Height):
        Outter_nodes_indicies=[]

        for i ,node in enumerate(Nodes):
            if (node[0]==Width or node[0]==0.0):
                Outter_nodes_indicies.append(i)

            elif (node[1]==0.0 or node[1]==Height)and(node [0]>0.0 and node[0]<Width):
                Outter_nodes_indicies.append(i)

        return np.array(Outter_nodes_indicies)

    @staticmethod
    def Generate_point_source_node(Width, Height, Nx, Ny, Nodes, Topeni_x, Topeni_y, zed):
        key_x = zed + Topeni_x / 2
        key_y = zed + Topeni_y / 2
        source_node_idx = 0
        dx=(Width/(Nx-1))*0.9
        dy=(Height/(Ny-1))*0.9
        for i, node in enumerate(Nodes):
            if np.isclose(node[0], key_x, atol=dx) and np.isclose(node[1], key_y, atol=dy):
                source_node_idx = i
        print(Nodes[source_node_idx][0], Nodes[source_node_idx][1])
        return source_node_idx

    @staticmethod
    def Generate_edges_couples_for_Newton(Nodes ,L_x, L_y, N_x, N_y):


        at_edge_1= np.argwhere((Nodes[:,1]==0.0 )).reshape(N_x)
        at_edge_2 = np.argwhere((Nodes[:, 1] == L_y)).reshape(N_x)
        at_edge_3 = np.argwhere((Nodes[:, 0] == 0.0)).reshape(N_y)
        at_edge_4 = np.argwhere((Nodes[:, 0] == L_x)).reshape(N_y)
        edge_1=Nodes[at_edge_1]
        edge_2=Nodes[at_edge_2]
        edge_3 = Nodes[at_edge_3]
        edge_4 = Nodes[at_edge_4]

        Couples_Edge_1= np.zeros((N_x-1,2))
        Couples_Edge_2 = np.zeros((N_x-1,2))
        Couples_Edge_3 = np.zeros((N_y-1,2))
        Couples_Edge_4 = np.zeros((N_y-1,2))
        k=0
        for i in range(1,N_x):
            Couples_Edge_1[k]= np.array([at_edge_1[i-1],at_edge_1[i]])
            Couples_Edge_2[k] = np.array([at_edge_2[i - 1], at_edge_2[i]])
            k+=1
        k=0
        for i in range(1, N_y):
            Couples_Edge_3[k] = np.array([at_edge_3[i - 1], at_edge_3[i]])
            Couples_Edge_4[k] = np.array([at_edge_4[i - 1], at_edge_4[i]])
            k+=1
        return np.array([Couples_Edge_1, Couples_Edge_2], dtype=int), np.array([Couples_Edge_3, Couples_Edge_4], dtype=int)





    #@staticmethod
    #def GenerateEdges_for_Newton():



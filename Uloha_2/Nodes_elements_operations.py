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
        return np.array([id_point_1,id_point_2,id_point_3, id_point_4]).flatten()

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

    @staticmethod
    def Generate_laser_Source_node( Nodes, L_x,L_y, N_x, N_y):
        id_point_1 = np.argwhere(
            (np.isclose(Nodes[:, 0], 0, atol=0.9 * (L_x / (N_x - 1)))) &
            (np.isclose(Nodes[:, 1], 0, atol=0.9 * (L_y / (N_y - 1))))
        )
        return id_point_1

    @staticmethod
    def Generate_Source(Nodes, l_x_source, l_y_source, id_mid):
        Mid = Nodes[id_mid]

        source_x_max, source_x_min = Mid[0] + l_x_source / 2, Mid[0] - l_x_source / 2
        source_y_max, source_y_min = Mid[1] + l_y_source / 2, Mid[1] - l_y_source / 2

        # Automatická tolerance (např. 1 % z délky zdroje, aby to trefilo uzly)
        atolx = l_x_source * 0.01
        atoly = l_y_source * 0.01
        is_in_x_1,is_in_x_2 = (np.isclose(Nodes[:, 0], source_x_max, atol=atolx)),(np.isclose(Nodes[:, 0], source_x_min,atol=atolx))
        is_in_y_edge =((source_y_min-atoly)<= Nodes[:,1])& (Nodes[:,1]<=(source_y_max+atoly))
        is_in_x_edge = (source_x_min-atolx<= Nodes[:,0]) &(Nodes[:,0]<=(source_x_max+atolx))

        is_in_y_1, is_in_y_2=(np.isclose(Nodes[:, 1], source_y_max, atol=atoly)), (np.isclose(Nodes[:, 1], source_y_min,atol=atoly))
        is_in_edge_1=is_in_x_1 & is_in_y_edge
        is_in_edge_2=is_in_x_2 & is_in_y_edge
        is_in_edge_3=is_in_y_1 & is_in_x_edge
        is_in_edge_4=is_in_y_2 & is_in_x_edge

        id_edge_1=np.argwhere(is_in_edge_1).flatten()
        id_edge_2=np.argwhere(is_in_edge_2).flatten()
        id_edge_3=np.argwhere(is_in_edge_3).flatten()
        id_edge_4=np.argwhere(is_in_edge_4).flatten()

        source_nodes = np.unique(
            np.concatenate([
                id_edge_1,
                id_edge_2,
                id_edge_3,
                id_edge_4
            ])
        )

        number_of_source_nodes = len(source_nodes)


        couples_edge_1= np.zeros((len(id_edge_1)-1,2))
        couples_edge_2 = np.zeros((len(id_edge_2) - 1, 2))
        couples_edge_3 = np.zeros((len(id_edge_3) - 1, 2))
        couples_edge_4 = np.zeros((len(id_edge_4) - 1, 2))
        k=0
        for i in range(1,id_edge_1.size):
            id1,id2=id_edge_1[i-1],id_edge_1[i]
            id3,id4=id_edge_2[i-1],id_edge_2[i]
            couples_edge_1[k]=np.array([id1,id2])
            couples_edge_2[k]=np.array([id3,id4])
            k+=1
        k=0
        for i in range(1,id_edge_3.size):
            id1, id2 = id_edge_3[i - 1], id_edge_3[i]
            id3, id4 = id_edge_4[i - 1], id_edge_4[i]
            couples_edge_3[k] = np.array([id1,id2 ])
            couples_edge_4[k] = np.array([id3, id4])
            k+=1
        return np.array([couples_edge_1, couples_edge_2], dtype=int), np.array([couples_edge_3, couples_edge_4],dtype=int),

    #@staticmethod
    #def GenerateEdges_for_Newton():



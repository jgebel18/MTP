
import numpy as np
import matplotlib.pyplot as plt



class Common_methods_MTP:
    @staticmethod
    def Generateindicies( w,h):
        Indicies_inner= []
        Indicies_edge_u= []
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                if i >0 and i <h-1 and j>0 and j <w-1:
                    Indicies_inner.append(idx)
                else :
                    Indicies_edge_u.append(idx)
        return np.array(Indicies_inner),np.array(Indicies_edge_u)




    @staticmethod
    def GenerateNeigbours(Nodes, Inner_Nodes, Outer_Nodes,w, h):
        Nodes= Nodes.reshape(h,w)
        n_inner= (h-2)*(w-2)
        Q_matrix=np.zeros((n_inner,n_inner))
        R_matrix= np.zeros((n_inner,w*h-n_inner))

        for i in range(1,h-1):
             for j in range(1, w-1):
                    radek_matice = (i - 1) * (w - 2) + (j - 1)
                    sousedi= np.array([Nodes[i-1,j], Nodes[i+1,j], Nodes[i,j-1], Nodes[i,j+1]])
                    for neighbour in sousedi:
                        if neighbour in Inner_Nodes:
                            id_2= np.where(neighbour==Inner_Nodes)[0][0]
                            Q_matrix[radek_matice,id_2 ]=0.25
                        else:
                            id_2= np.where(neighbour==Outer_Nodes)
                            R_matrix[radek_matice,id_2 ]= 0.25
        return Q_matrix, R_matrix

    @staticmethod
    def IterationProcess(Q_matrix, Q_0):
        N_Exodus= np.zeros(Q_matrix.shape)
        Q=Q_0
        Steps=1000
        for i in range(Steps):
            N_Exodus+=Q
            Q=Q@Q_matrix
        return N_Exodus

    @staticmethod
    def GenerateThermalGradient(Outer_Nodes, T1, T2):
        T_out= np.ones(Outer_Nodes.shape)
        k=0
        for i, globalidx in enumerate(Outer_Nodes):
            #print(int(globalidx - k))
            if globalidx%11==0:
                T_out[i] = T1
            elif (int(globalidx-k))%10==0:
                T_out[i] = T2
                k+=1
            else:
                T_out[i] = (T1+T2)/2
        return T_out



    @staticmethod
    def GenretateFullGrid(T_boundaries, Outer_nodes, FullGrid):
        k=0
        for i in range(FullGrid.T[1].size):
            for j in range(FullGrid[0].size):
                if FullGrid[i][j]==Outer_nodes[k]:
                    FullGrid[i][j]=T_boundaries[k]
                    k+=1
        return FullGrid





    @staticmethod
    def PlotData(nazev, nazev_2, Nodes, T_plot,
                 coment_1 ,coment_2):
        plt.figure(figsize=(10, 10))
        plt.plot(Nodes, T_plot[2], )
        plt.grid(True)
        plt.title(coment_2)
        plt.xlabel(f'Vzdalenost v ose x [m]')
        plt.ylabel(f'Teplota [K]')
        plt.savefig(nazev_2)
        plt.show()

       #rozsah = [1, len(T_plot[3]), 1,  len(T_plot.T[3])]

        plt.figure(figsize=(10, 10))
        rozsah = [0, len(T_plot[4])-1, 0, len(T_plot.T[3])-1]
        plt.imshow(T_plot, cmap='jet', origin='lower',extent=rozsah, interpolation='bilinear')

        plt.colorbar(label='Teplota [K]')
        plt.title(coment_1)
        plt.xlabel('Vzdálenost v ose x [m]')
        plt.ylabel('Vzdálenost v ose y [m]')
        plt.savefig(nazev)
        plt.show()
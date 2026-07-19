import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri



class FEM_Methods:

    def __init__(self,Nodes,elements,
        Width, Height, Width_topeni, Height_topeni,
                 lamda_vzduch, lambda_zed, lambda_topeni, c_topeni, c_zed,
                 c_vzduch,rho_topeni, rho_zed, rho_vzduch ):
        self.elementsid= elements
        self.Nodes= Nodes
        self.Height = Height
        self.Width = Width
        self.Height_topeni = Height_topeni
        self.Width_topeni = Width_topeni
        self.lamda_vzduch = lamda_vzduch
        self.lambda_zed = lambda_zed
        self.lambda_topeni = lambda_topeni
        self.rho_topeni=rho_topeni
        self.rho_zed=rho_zed
        self.rho_vzduch=rho_vzduch
        self.c_topeni =c_topeni
        self.c_zed = c_zed
        self.c_vzduch =c_vzduch
        self.Global_Stiffness_matrix= None
        self.Global_mass_matrix= None
        self.F= None
        self.zed_tloustka=1.0

    def Generate_basics_functions(self, xi, eta):
        N_1= 1-xi-eta
        N_2= xi
        N_3= eta
        return np.array([N_1, N_2, N_3])

    def Generate_B_matrix(self, ):
        return np.array([[-1,1,0],[-1,0,1]])

    def Jacobian(self, element ):
        x_s = element[:,0] #np.array([x_1, x_2, x_3])
        y_s = element[:,1]#np.array([y_1, y_2, y_3])
        X= np.array([x_s,y_s])
        B=self.Generate_B_matrix()
        J= B@X.T
        return np.abs(np.linalg.det(J))

    def GenerateD_matrix(self):
        D_matrices = np.zeros((self.elementsid.T[0].size, 2, 2))

        for k, element in enumerate(self.elementsid):
            Element = self.Nodes[element]
            Mean_x = np.mean(Element[:, 0])
            Mean_y = np.mean(Element[:, 1])
            is_in_wall = (Mean_x < self.zed_tloustka) or \
                         (Mean_x > (self.Width - self.zed_tloustka)) or \
                         (Mean_y < self.zed_tloustka) or \
                         (Mean_y > (self.Height - self.zed_tloustka))
            if is_in_wall:
                D_matrices[k] = np.eye(2) * self.lambda_zed
            elif (self.zed_tloustka <= Mean_x <= (self.zed_tloustka + self.Width_topeni)) and \
                    (self.zed_tloustka <= Mean_y <= (self.zed_tloustka + self.Height_topeni)):
                D_matrices[k] = np.eye(2) * self.lambda_topeni
            else:
                D_matrices[k] = np.eye(2) * self.lamda_vzduch

        return D_matrices

    def GenerateD_matrix_time(self):
        D_matrices = np.zeros((self.elementsid.T[0].size, 2, 2))
        Rho_vector= np.zeros(self.elementsid.T[0].size)
        C_vector=np.zeros(self.elementsid.T[0].size)
        for k, element in enumerate(self.elementsid):
            Element = self.Nodes[element]
            Mean_x = np.mean(Element[:, 0])
            Mean_y = np.mean(Element[:, 1])
            is_in_wall = (Mean_x < self.zed_tloustka) or \
                         (Mean_x > (self.Width - self.zed_tloustka)) or \
                         (Mean_y < self.zed_tloustka) or \
                         (Mean_y > (self.Height - self.zed_tloustka))
            if is_in_wall:
                D_matrices[k] = np.eye(2) * self.lambda_zed
                Rho_vector[k]= self.rho_zed
                C_vector[k]= self.c_zed

            elif (self.zed_tloustka <= Mean_x <= (self.zed_tloustka + self.Width_topeni)) and \
                    (self.zed_tloustka <= Mean_y <= (self.zed_tloustka + self.Height_topeni)):
                D_matrices[k] = np.eye(2) * self.lambda_topeni
                Rho_vector[k]= self.rho_topeni
                C_vector[k]=self.c_topeni
            else:
                D_matrices[k] = np.eye(2) * self.lamda_vzduch
                Rho_vector[k]= self.rho_vzduch
                C_vector[k]=self.c_vzduch

        return D_matrices,Rho_vector, C_vector

    def Generate_local_matrices_time(self):
        D_matrices, Rho_vector, C_vector = self.GenerateD_matrix_time()
        K = np.zeros((len(self.elementsid), 3, 3))
        M=np.zeros((len(self.elementsid), 3, 3))
        basics_element_matrix= np.array([[2,1,1],[1,2,1],[1,1,2]])
        for k, element in enumerate(self.elementsid):
            Element = self.Nodes[element]
            B = self.Generate_B_matrix()
            x = Element[:, 0]
            y = Element[:, 1]
            J = B @ np.array([x, y]).T
            detJ = abs(np.linalg.det(J))
            B_global = np.linalg.inv(J) @ B
            D = D_matrices[k]
            K[k] = 0.5 * detJ * (B_global.T @ D @ B_global)
            M[k] =(1/24)* detJ * Rho_vector[k]*C_vector[k]*basics_element_matrix
        return K, M

    def Generate_local_Stiffness_matrix(self):
        D_matrices = self.GenerateD_matrix()
        K = np.zeros((len(self.elementsid), 3, 3))

        for k, element in enumerate(self.elementsid):
            Element = self.Nodes[element]
            B = self.Generate_B_matrix()
            x = Element[:, 0]
            y = Element[:, 1]
            J = B @ np.array([x, y]).T
            detJ = abs(np.linalg.det(J))
            B_global = np.linalg.inv(J) @ B
            D = D_matrices[k]
            K[k] = 0.5 * detJ * (B_global.T @ D @ B_global)

        return K



    def Generate_Global_Stiffness_matrix(self):
        Global_k= np.zeros((self.Nodes.T[0].size,self.Nodes.T[0].size))
        K_s= self.Generate_local_Stiffness_matrix()
        k=0
        for element in self.elementsid:
            Element= self.Nodes[element]
            K= K_s[k]
            for i, idx in enumerate(element):
                for j, jdx in enumerate(element):
                    global_i= idx
                    global_j= jdx
                    Global_k[global_i,global_j]+= K[i,j]
            k+=1
        self.Global_Stiffness_matrix= Global_k
        return Global_k

    def Generate_Global_Matrices_Time(self):
        Global_k= np.zeros((self.Nodes.T[0].size,self.Nodes.T[0].size))
        Global_m = np.zeros((self.Nodes.T[0].size,self.Nodes.T[0].size))
        K_s,M_s= self.Generate_local_matrices_time()
        k=0
        for element in self.elementsid:
            Element= self.Nodes[element]
            K= K_s[k]
            m=M_s[k]
            for i, idx in enumerate(element):
                for j, jdx in enumerate(element):
                    global_i= idx
                    global_j= jdx
                    Global_k[global_i,global_j]+= K[i,j]
                    Global_m[global_i,global_j]+= m[i,j]
            k+=1
        self.Global_Stiffness_matrix= Global_k
        self.Global_mass_matrix= Global_m
        return Global_k,Global_m



import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.differentiate import derivative
from scipy.sparse import csr_matrix



class FEM_Methods:

    def __init__(self,Nodes,elements,
        Width, Height, Width_topeni, Height_topeni,
                 lamda_vzduch, lambda_zed, lambda_topeni, c_topeni, c_zed,
                 c_vzduch,rho_topeni, rho_zed, rho_vzduch, velocity  ):
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
        self.velocity= velocity

    def Generate_basics_functions(self, xi, eta):
        N_1= 1-xi-eta
        N_2= xi
        N_3= eta
        return np.array([N_1, N_2, N_3])

    def Generate_Velocity_vector(self, x,y):
        h=1e-6
        v_x= -(1/h)*(self.Generate_jet_function(x, y+h)-self.Generate_jet_function(x,y))
        v_y= (1/h)*(self.Generate_jet_function(x+h, y)-self.Generate_jet_function(x,y))
        return np.array([v_x,v_y])



    def Generate_jet_function(self, x,y):
        L_x= self.Width-2*self.zed_tloustka
        L_y= self.Height-2*self.zed_tloustka
        sigma= 2*0.8**2
        result_x= (np.sin(np.pi*(x-self.zed_tloustka)/L_x))**2
        result_y= (np.sin(np.pi*(y-self.zed_tloustka)/L_y))**2
        Gauss= np.exp((-(x-2)**2)/sigma)
        return self.velocity*result_x*result_y*Gauss


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


    def Generate_matrices_jet(self):
        D_matrices = np.zeros((self.elementsid.T[0].size, 2, 2))
        Velocity_vector=np.zeros((self.elementsid.T[0].size, 2))
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

                Velocity_vector[k]= np.array([0.0, 0.0])

            elif (self.zed_tloustka <= Mean_x <= (self.zed_tloustka + self.Width_topeni)) and \
                    (self.zed_tloustka <= Mean_y <= (self.zed_tloustka + self.Height_topeni)):
                D_matrices[k] = np.eye(2) * self.lambda_topeni

                Velocity_vector[k]=np.array([0.0,0.0])
            else:
                D_matrices[k] = np.eye(2) * self.lamda_vzduch

                Velocity_vector[k]=self.Generate_Velocity_vector(Mean_x, Mean_y)
        return D_matrices,Velocity_vector


    def Generate_matrices_time_jet(self):
        D_matrices = np.zeros((self.elementsid.T[0].size, 2, 2))
        Rho_vector= np.zeros(self.elementsid.T[0].size)
        C_vector=np.zeros(self.elementsid.T[0].size)
        Velocity_vector=np.zeros((self.elementsid.T[0].size, 2))
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
                Velocity_vector[k]= np.array([0.0, 0.0])

            elif (self.zed_tloustka <= Mean_x <= (self.zed_tloustka + self.Width_topeni)) and \
                    (self.zed_tloustka <= Mean_y <= (self.zed_tloustka + self.Height_topeni)):
                D_matrices[k] = np.eye(2) * self.lambda_topeni
                Rho_vector[k]= self.rho_topeni
                C_vector[k]=self.c_topeni
                Velocity_vector[k]=np.array([0.0,0.0])
            else:
                D_matrices[k] = np.eye(2) * self.lamda_vzduch
                Rho_vector[k]= self.rho_vzduch
                C_vector[k]=self.c_vzduch
                Velocity_vector[k]=self.Generate_Velocity_vector(Mean_x, Mean_y)
        return D_matrices,Rho_vector, C_vector,Velocity_vector




    def GenerateD_matrix_time(self):
        D_matrices = np.zeros((self.elementsid.T[0].size, 2, 2))
        Rho_vector = np.zeros(self.elementsid.T[0].size)
        C_vector = np.zeros(self.elementsid.T[0].size)
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
                Rho_vector[k] = self.rho_zed
                C_vector[k] = self.c_zed

            elif (self.zed_tloustka <= Mean_x <= (self.zed_tloustka + self.Width_topeni)) and \
                    (self.zed_tloustka <= Mean_y <= (self.zed_tloustka + self.Height_topeni)):
                D_matrices[k] = np.eye(2) * self.lambda_topeni
                Rho_vector[k] = self.rho_topeni
                C_vector[k] = self.c_topeni
            else:
                D_matrices[k] = np.eye(2) * self.lamda_vzduch
                Rho_vector[k] = self.rho_vzduch
                C_vector[k] = self.c_vzduch

        return D_matrices, Rho_vector, C_vector

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


    def Generate_local_matrices_time_jet(self):
        D_matrices, Rho_vector, C_vector, Velocites  = self.Generate_matrices_time_jet()
        K=np.zeros((len(self.elementsid), 3, 3))
        K_stiff = np.zeros((len(self.elementsid), 3, 3))
        M=np.zeros((len(self.elementsid), 3, 3))
        K_adv= K.copy()
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
            K_stiff[k] = 0.5 * detJ * (B_global.T @ D @ B_global)
            M[k] =(1/24)* detJ * Rho_vector[k]*C_vector[k]*basics_element_matrix
            a=(Velocites[k].T @ B_global)
            A = np.outer(np.ones(3), a)
            K_adv[k]= (detJ/6)*C_vector[k]*Rho_vector[k]*A
            K[k]= K_stiff[k] + K_adv[k]
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

    def Generate_local_Stiffness_matrix_Jet(self):
        D_matrices, Rho_vector, C_vector, Velocites  = self.Generate_matrices_time_jet()
        K = np.zeros((len(self.elementsid), 3, 3))
        K_stiff = np.zeros((len(self.elementsid), 3, 3))

        K_adv = K.copy()

        for k, element in enumerate(self.elementsid):
            Element = self.Nodes[element]
            B = self.Generate_B_matrix()
            x = Element[:, 0]
            y = Element[:, 1]
            J = B @ np.array([x, y]).T
            detJ = abs(np.linalg.det(J))
            B_global = np.linalg.inv(J) @ B
            D = D_matrices[k]
            K_stiff[k] = 0.5 * detJ * (B_global.T @ D @ B_global)

            a = (Velocites[k].T @ B_global)
            A = np.outer(np.ones(3), a)
            K_adv[k] = (detJ / 6) * C_vector[k] * Rho_vector[k] * A
            K[k] = K_stiff[k] + K_adv[k]
        return K

    def Generate_velocity(self, x, y):
        L_x=self.Width-self.zed_tloustka
        L_y=self.Height-self.zed_tloustka
        v_x= (self.velocity*np.sin((np.pi*(x+(L_x+self.zed_tloustka)/2-(self.Width_topeni+self.zed_tloustka))/(1.5*(L_x+self.zed_tloustka)-(self.Width_topeni+self.zed_tloustka))))*
              np.cos((np.pi*(y-(self.Height_topeni+self.zed_tloustka)))/((L_y+self.zed_tloustka)-(self.Height_topeni+self.zed_tloustka))))
        v_y= 0.0
        return np.array([v_x, v_y])








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
        return csr_matrix(Global_k)

    def Generate_Global_Stiffness_matrix_Jet(self):
        Global_k= np.zeros((self.Nodes.T[0].size,self.Nodes.T[0].size))
        K_s= self.Generate_local_Stiffness_matrix_Jet()
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
        return csr_matrix(Global_k)


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
        return csr_matrix(Global_k),csr_matrix(Global_m)


    def Generate_Global_Matrices_Time_jet(self):
        Global_k= np.zeros((self.Nodes.T[0].size,self.Nodes.T[0].size))
        Global_m = np.zeros((self.Nodes.T[0].size,self.Nodes.T[0].size))
        K_s,M_s= self.Generate_local_matrices_time_jet()
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
        return csr_matrix(Global_k),csr_matrix(Global_m)

    def Generate_Nodes_of_air(self):
        Nodes_air = np.zeros_like(self.Nodes)

        for k, node in enumerate(self.Nodes):
            x, y = node[0], node[1]

            # 1. Kontrola, zda je uzel ve zdi (včetně vnější hranice)
            is_in_wall = (x <= self.zed_tloustka) or \
                         (x >= (self.Width - self.zed_tloustka)) or \
                         (y <= self.zed_tloustka) or \
                         (y >= (self.Height - self.zed_tloustka))

            # 2. Kontrola, zda je uzel ČISTĚ UVNITŘ topení (stěny zůstanou pro vzduch)
            is_in_heater_interior = (self.zed_tloustka < x < (self.zed_tloustka + self.Width_topeni)) and \
                                    (self.zed_tloustka < y < (self.zed_tloustka + self.Height_topeni))

            # Pokud není ve zdi ani uvnitř topení, je to vzduch (včetně povrchu topidla)
            if not is_in_wall and not is_in_heater_interior:
                Nodes_air[k] = node
            else:
                Nodes_air[k] = np.array([np.nan, np.nan])

        return Nodes_air

    def Plot_Nodes_of_air(self):
        Nodes_air = self.Generate_Nodes_of_air()

        plt.figure(figsize=(10, 8))

        # vykreslení pouze platných uzlů
        valid_nodes = ~np.isnan(Nodes_air[:, 0])

        plt.scatter(
            Nodes_air[valid_nodes, 0],
            Nodes_air[valid_nodes, 1],
            s=10
        )

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Uzly ve vzduchu")
        plt.axis("equal")
        plt.grid()

        plt.show()

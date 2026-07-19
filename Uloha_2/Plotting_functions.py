import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np




class PlottingFunctions:
    @staticmethod
    def Plot_Thermal_mapping(tri ,T, Nodes ,BC, Element_view):
        plt.figure(figsize=(12, 8))
        # Vykreslení teplotního pole na trojúhelníkové síti
        tcf = plt.tricontourf(tri, T, levels=100, cmap='jet')
        cb = plt.colorbar(tcf)
        cb.set_label('Teplota [°C]', fontsize=12)
        triang = mtri.Triangulation(Nodes[:, 0], Nodes[:, 1])
        # Vykreslení geometrických obrysů pro kontrolu
        plt.plot([0, 10, 10, 0, 0], [0, 0, 8, 8, 0], 'k-', linewidth=2.5, label='Vnější stěna')
        plt.plot([1, 9, 9, 1, 1], [1, 1, 7, 7, 1], 'k--', linewidth=1.5, label='Vnitřní líc zdi (1m)')
        plt.plot([1, 3, 3, 1, 1], [1, 1, 2, 2, 1], 'g-', linewidth=2, label='Topidlo')
        # Příkaz triplot vykreslí samotné čáry trojúhelníků
        # 'g-' znamená zelená (green) plná čára. Můžeš dát 'b-' pro modrou, 'k-' pro černou atd.
        if Element_view==True:
            plt.triplot(triang, 'g-', linewidth=0.5, alpha=0.7)
        else:
            pass
        if (BC==f'Dirichlet'):
            plt.title('Stacionární teplotní pole s Dirichletovou okrajovou podmínkou', fontsize=14, fontweight='bold')
        elif(BC=='Newton'):
            plt.title('Stacionární teplotní pole s Newtonovou okrajovou podmínkou', fontsize=14, fontweight='bold')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        if (BC == f'Dirichlet'):
            plt.savefig(f'Thermal_mapping_Dirichlet_Sationar.png')
        elif (BC == 'Newton'):
            plt.savefig(f'Thermal_mapping_Newton_Sationar.png')
        plt.show()


    @staticmethod
    def Plot_Thermal_mapping_time(tri ,T, Nodes , Element_view, time):
        plt.figure(figsize=(12, 8))
        # Vykreslení teplotního pole na trojúhelníkové síti
        tcf = plt.tricontourf(tri, T, levels=100, cmap='jet')
        cb = plt.colorbar(tcf)
        cb.set_label('Teplota [°C]', fontsize=12)
        triang = mtri.Triangulation(Nodes[:, 0], Nodes[:, 1])
        # Vykreslení geometrických obrysů pro kontrolu
        plt.plot([0, 10, 10, 0, 0], [0, 0, 8, 8, 0], 'k-', linewidth=2.5, label='Vnější stěna')
        plt.plot([1, 9, 9, 1, 1], [1, 1, 7, 7, 1], 'k--', linewidth=1.5, label='Vnitřní líc zdi (1m)')
        plt.plot([1, 3, 3, 1, 1], [1, 1, 2, 2, 1], 'g-', linewidth=2, label='Topidlo')
        # Příkaz triplot vykreslí samotné čáry trojúhelníků
        # 'g-' znamená zelená (green) plná čára. Můžeš dát 'b-' pro modrou, 'k-' pro černou atd.
        if Element_view==True:
            plt.triplot(triang, 'g-', linewidth=0.5, alpha=0.7)
        else:
            pass

        plt.title(f'Teplotní pole Newtonovy okrajové úlohy pro čas : {np.round(time/86400, 2)} dní', fontsize=14, fontweight='bold')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis('equal')
        plt.grid(True, linestyle=':', alpha=0.5)

        plt.legend(loc='upper right')


        plt.savefig(f'Thermal_mapping_Newton_Sationar-time-{time}.png')
        plt.show()

    @staticmethod
    def Plot_Triagonal (Nodes ):
        plt.figure(figsize=(12, 8))
        # Vykreslení teplotního pole na trojúhelníkové síti


        triang = mtri.Triangulation(Nodes[:, 0], Nodes[:, 1])
        # Vykreslení geometrických obrysů pro kontrolu
        plt.plot([0, 10, 10, 0, 0], [0, 0, 8, 8, 0], 'k-', linewidth=2.5, label='Vnější stěna')
        plt.plot([1, 9, 9, 1, 1], [1, 1, 7, 7, 1], 'k--', linewidth=1.5, label='Vnitřní líc zdi (1m)')
        plt.plot([1, 3, 3, 1, 1], [1, 1, 2, 2, 1], 'g-', linewidth=2, label='Topidlo')


        plt.triplot(triang, 'g-', linewidth=0.5, alpha=0.7)
        plt.title(f'Ukázka prvkové sítě na simulovaném objektu')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')

        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        plt.savefig('Diskretizační_schéma.png')
        plt.show()


    @staticmethod
    def GenerateThermalprofile_stationar( Nx,Nodes,T, BC,indicies):
        Modified_nodes= Nodes[indicies].reshape(Nx,2)
        Thermal_modified= T[indicies]
        plt.figure(figsize=(12, 8))

        plt.scatter(Modified_nodes[:,0],Thermal_modified ,marker="o")
        plt.xlabel('X [m]')

        plt.ylabel(' T[°C] ')
        if (BC == f'Dirichlet'):
            plt.title(f'Teplotní profil stacionární  Dirichletovy úlohy',
                      fontsize=14, fontweight='bold')
        elif (BC == 'Newton'):
            plt.title(f'Teplotní profil stacionární Newtonovy úlohy',
                      fontsize=14, fontweight='bold')

        plt.grid(True, linestyle=':', alpha=0.5)
        if (BC == f'Dirichlet'):
            plt.savefig(f'Primka_Dirichlet_Sationar.png')
        elif (BC == 'Newton'):
            plt.savefig(f'Primka_Newton_Sationar.png')
        plt.show()
    @staticmethod
    def GenerateThermalProfileUnstationar( Nx,Nodes,T,indicies, Time):
        Modified_nodes = Nodes[indicies]
        Thermal_modified = T[indicies]
        plt.figure(figsize=(12, 8))

        plt.scatter(Modified_nodes[:, 0], Thermal_modified, marker="o")
        plt.xlabel('X [m]')

        plt.ylabel(' T[°C] ')

        plt.title(f'Teplotní profil Newtonovy úlohy pro čas: {np.round(Time/86400, 2)}',
                  fontsize=14, fontweight='bold')

        plt.grid(True, linestyle=':', alpha=0.5)

        plt.savefig(f'Primka_Newton_Sationar_time_{Time}.png')

        plt.show()

    @staticmethod
    def Generate_Time_Thermal_Dependence(T_vector, indicies, Nodes, Time):

        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(indicies):
            # Opraveno formátování popisku (přidány mezery kolem dvojteček pro lepší čitelnost)
            plt.plot(Time / 86400, T_vector[i], label=f'Bod: {i + 1}')

        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel('Čas [dny]')  # Volitelně můžeš nechat 'Time [days]', pokud píšeš grafy anglicky
        plt.ylabel('T [°C]')
        plt.legend(loc='upper right')

        # Gramatické opravy v titulku:
        # 1. "teploty" s malým "t"
        # 2. "Newtonovy" (Newtonova úloha, tvrdé y na konci podle vzoru mladá, s "w")
        plt.title('Časová závislost teploty v zadaných bodech v rámci Newtonovy nestacionární úlohy',
                  fontsize=14, fontweight='bold')

        plt.savefig('Time_dependence.png')
        plt.show()


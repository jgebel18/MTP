import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Síť
# --------------------------------
Lx = 2.0
Ly = 1.0

Nx = 40
Ny = 20

dx = Lx / Nx
dy = Ly / Ny

rho = 1.0
u = 1.0
v = 0.0
Gamma = 0.01

# --------------------------------
# Pole phi ve středech objemů
# --------------------------------
phi = np.zeros((Ny, Nx))

# levá okrajová podmínka
phi[:, 0] = 1.0

# --------------------------------
# Iterační řešení
# --------------------------------
for iteration in range(5000):

    phi_old = phi.copy()

    for j in range(Ny):
        for i in range(Nx):

            # levá stěna
            if i == 0:
                phi_W = 1.0
            else:
                phi_W = phi[j, i-1]

            # pravá stěna
            if i == Nx - 1:
                phi_E = 0.0
            else:
                phi_E = phi[j, i+1]

            # dolní stěna
            if j == 0:
                phi_S = phi[j, i]
            else:
                phi_S = phi[j-1, i]

            # horní stěna
            if j == Ny - 1:
                phi_N = phi[j, i]
            else:
                phi_N = phi[j+1, i]

            # difuzní koeficienty
            Dw = Gamma * dy / dx
            De = Gamma * dy / dx
            Ds = Gamma * dx / dy
            Dn = Gamma * dx / dy

            # konvekce
            Fw = rho * u * dy
            Fe = rho * u * dy

            # Upwind schéma
            aW = Dw + max(Fw, 0)
            aE = De + max(-Fe, 0)
            aS = Ds
            aN = Dn

            aP = aW + aE + aS + aN

            # pravá strana
            b = 0.0

            # levá Dirichletova podmínka
            if i == 0:
                b += aW * 1.0

            # aktualizace
            phi[j, i] = (
                aW * phi_W +
                aE * phi_E +
                aS * phi_S +
                aN * phi_N +
                b
            ) / aP

    # kontrola konvergence
    error = np.max(np.abs(phi - phi_old))

    if error < 1e-6:
        print("Konvergence:", iteration)
        break

# --------------------------------
# Graf
# --------------------------------
X = np.linspace(dx/2, Lx-dx/2, Nx)
Y = np.linspace(dy/2, Ly-dy/2, Ny)

X, Y = np.meshgrid(X, Y)

plt.figure(figsize=(10, 4))
plt.contourf(X, Y, phi, levels=30)
plt.colorbar(label="phi")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
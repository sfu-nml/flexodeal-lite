@page 01-introduction-to-flexodeal Introduction to Flexodeal
@tableofcontents

This tutorial depends on deal.II's [step-44](https://www.dealii.org/current/doxygen/deal.II/step_44.html). Unless otherwise specified, all phyisical quantities have SI units.

# Mathematical formulation

Consider a muscular structure (e.g. a block of muscle tissue) which, at time \f$t \geq 0\f$, occupies a region \f$\mathcal{B}_t \subset \mathbb{R}^3\f$. Given an activation profile \f$ a = a(t) \f$ and a boundary strain \f$ \varepsilon = \varepsilon(t) \f$, we wish to compute a displacement field \f$\mathbf{U}(\mathbf{X},t)\f$, a pressure field \f$p(\mathbf{X},t)\f$, and a dilation field \f$D(\mathbf{X},t)\f$ such that
\f[
\begin{gathered}
\rho_0 \mathbf{U}_{tt} = \mathbf{Div}  \ \mathbf{P(a,\mathbf{U},\mathbf{U}_t)} + \mathbf{f}_0 \quad \text{in } \mathcal{B}_0 \times (0,\infty), \\
J(U) - D = 0 \quad \text{in } \mathcal{B}_0 \times (0,\infty), \\
p - \Psi_{vol}'(D) = 0 \quad \text{in } \mathcal{B}_0 \times (0,\infty), \\
\mathbf{U} = \varepsilon(\cdot) L_0 \hat{\mathbf{i}} \quad \text{on } \Gamma_{0,D} \times (0,\infty), \\
\mathbf{P(a,\mathbf{U},\mathbf{U}_t)} \mathbf{N} = \mathbf{0} \quad \text{on } \Gamma_{0,N} \times (0,\infty), \\
\mathbf{U} = \mathbf{0}, \quad p = 0, \quad D = 1 \quad \text{at } t=0.
\end{gathered}
\f]
Here, \f$\rho_0\f$ is the initial tissue density, \f$\mathbf{f}_0\f$ is a volumetric force per unit volume, \f$\Gamma_{0,D}\f$ represents boundary faces on which we impose strains, and \f$\Gamma_{0,N}\f$ is the collection of traction-free boundary faces. In addition, \f$\Psi_{vol}\f$ is the volumetric energy of the system given by
\f[
    \Psi_{vol}(D) = \dfrac{\kappa}{4}\left( D^2 - 2 \ln D - 1 \right),
\f]
where $\kappa$ is the bulk modulus of muscle tissue.

## Description of the stress tensor

The formulation described above is based on a volumetric-isochoric decomposition of the problem. In particular, the strain energy of the system can be thought as\f$\Psi = \Psi_{vol} + \Psi_{iso} \f$, although no explicit expression exists for \f$\Psi_{iso}\f$. The first Piola-Kirchhoff tensor is best described using the second 
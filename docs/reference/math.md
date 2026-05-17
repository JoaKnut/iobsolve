# Fundamentos Matemáticos

## Dominio Continuo (Paper I)

### El Operador de Integridad de Bisagra

Sea $\Omega \subset \mathbb{R}^n$ una variedad euclidiana compacta equipada con
un campo diferenciable escalar (o vectorial) $\phi : \Omega \to \mathbb{R}^m$.
El **Operador de Integridad de Bisagra** (IOB) cuantifica la curvatura geométrica local:

$$\mathcal{H}(x) = \left| \nabla^2 \phi(x) \right|$$

calculado mediante diferencias finitas centrales de segundo orden:

$$\nabla^2 \phi \approx \sum_{i=1}^n \frac{\phi(x + h\,\mathbf{e}_i) - 2\phi(x) + \phi(x - h\,\mathbf{e}_i)}{h^2}$$

### Aislamiento de Raíces (Teorema del Flujo de Integridad)

Dado un campo vectorial $F : \mathbb{R}^n \to \mathbb{R}^n$, el IOB-QuadTree localiza
subdominios $\Omega_k \subset \Omega$ que satisfacen dos criterios simultáneos:

**Criterio 1 — TVI (Topológico):**
$$\min_{\Omega_k} F_c \leq 0 \leq \max_{\Omega_k} F_c \quad \forall c = 1,\ldots,n$$

**Criterio 2 — IOB-FFT (Espectral):**
$$\mathcal{Q}_\text{spec} = \frac{\sum_{\nu > \nu_c} \sum_c |\hat{F}_c(\nu)|^2}
                                  {\sum_\nu \sum_c |\hat{F}_c(\nu)|^2} > \tau_c$$

Solo los subdominios que satisfacen ambos criterios se bisectan adicionalmente, logrando
una localización asintóticamente acelerada frente al Newton-Raphson clásico.

---

## Dominio Discreto (Paper II)

### Laplacianos de Grafos

Para un grafo ponderado $\mathcal{G}(V, E, W)$ con matriz de adyacencia
$\mathbf{W} \in \mathbb{R}^{N\times N}$ y matriz de grados
$\mathbf{D}_{ii} = \sum_j W_{ij}$:

**Laplaciano Combinatorio:**
$$\mathbf{L} = \mathbf{D} - \mathbf{W}$$

**Laplaciano Normalizado (simétrico):**
$$\mathcal{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$$

Propiedades espectrales:
- $\mathbf{L}$ es semidefinido positivo; autovalores $\in [0, \infty)$.
- $\mathcal{L}$ tiene autovalores en $[0, 2]$.
- El vector de unos $\mathbf{1}$ siempre está en el núcleo: $\mathbf{L}\mathbf{1} = \mathbf{0}$.

### Índice de Estrés Nodal

El índice de estrés nodal $Q_i$ es la magnitud normalizada del residuo baricéntrico:

$$\mathbf{R}(t) = -\mathbf{L}\,\mathbf{X}(t)$$
$$Q_i(t) = \frac{\|\mathbf{R}_i(t)\|_2}{\max_j \|\mathbf{R}_j(t)\|_2 + \varepsilon}$$

### Z-Score Topológico Robusto

El Z-Score modificado basado en MAD (Desviación Absoluta de la Mediana):

$$\mathcal{M}_i(t) = \frac{0.6745\,(Q_i - \tilde{Q}^*(t))}{\max(\text{MAD}(t),\;\varepsilon)}$$

donde $\tilde{Q}^*(t) = \lambda\,\tilde{Q}(t-1) + (1-\lambda)\,\text{mediana}(Q(t))$
es la mediana mezclada exponencialmente. La constante $0.6745$ asegura consistencia
con $\sigma$ bajo distribuciones gaussianas: $\text{MAD}[\mathcal{N}(0,\sigma)] = 0.6745\,\sigma$.

**Punto de ruptura estadístico**: 50 % (frente al ~0 % de los Z-Scores basados en media/std bajo ataques de enmascaramiento).

### Cirugía Topológica

Al detectar un nodo anómalo $v^*$ con $\mathcal{M}_{v^*} > \tau$, la
adyacencia se actualiza:

$$W_{v^*,j}(t^+) = W_{j,v^*}(t^+) = 0 \quad \forall j \in V$$

Complejidad en Sparse COO: $\mathcal{O}(k_{v^*})$ (máscara sobre los arrays de coordenadas activas).

---

## Referencias Clave

1. Knuttzen, J. (2026). *Formalismo de Integridad de Bisagra: Aislamiento Topológico de Singularidades y Control de Bifurcaciones en Variedades Continuas.*
2. Knuttzen, J. (2026). *Formalismo de Integridad de Bisagra Discreto: Laplacianos de Grafos, Detección de Anomalías Asíncronas y Colapsos en Redes Complejas.*
3. Chung, F.R.K. (1997). *Spectral Graph Theory.* AMS.
4. Leys, C. et al. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *Journal of Experimental Social Psychology*, 49(4), 764–766.
5. Lorenz, E.N. (1996). Predictability — A problem partly solved. *Seminario ECMWF sobre Predictabilidad.*

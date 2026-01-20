# Tardigrade Optimization Algorithm (TOA)

The **Tardigrade Optimization Algorithm (TOA)** is a contemporary **bio-inspired metaheuristic** optimization technique modeled after the adaptable survival behavior of **tardigrades**, microscopic organisms capable of enduring extreme environmental conditions through states like **cryptobiosis** and controlled **energy conservation**.

This algorithm integrates **exploration–exploitation balancing**, **hunger-driven adaptation**, and **cooperative movement modeling** to provide a versatile and robust optimization framework.  




## Algorithm Overview

TOA emulates five major biological behaviors:


### 1. Hunger-Driven Movement Mechanics

Tardigrades regulate mobility based on resource availability.  
In TOA, each agent’s movement intensity depends on its **hunger level** \( H_i \), which adaptively changes according to its performance.

The hunger update follows:

$$
H_i(t+1) =
\begin{cases}
H_i(t) - \delta_h, & \text{if } f(\mathbf{x}_i) < \bar{f}(t) \\
H_i(t) + \delta_p, & \text{otherwise}
\end{cases}
$$



Lower hunger reduces activity, encouraging fine exploitation; higher hunger induces active searching.

---

### 2. Cryptobiosis Modeling

When hunger surpasses a threshold \( H_i > H_c \), the agent enters a reduced-mobility state.  
Its position update becomes minimal or frozen:

$$
\mathbf{x}_i(t+1) =
\mathbf{x}_i(t) + \epsilon \cdot \mathcal{U}(-\eta, \eta)
$$



### 3. Elite-Based Learning and Cooperation

A subset of **elite agents** \( E \subset N \) are selected based on best fitness values.

Cooperation occurs either with the **global best** or a **random elite** member:

$$
\mathbf{x}_i(t+1) =
\mathbf{x}_i(t) +
\lambda \cdot H_i(t) \cdot (\mathbf{x}_E - \mathbf{x}_i(t))
$$



### 4. Lévy-Flight-Assisted Exploration

To ensure global diversity, some agents apply **Lévy flight** steps:

$$
\mathbf{x}_i(t+1) = \mathbf{x}_i(t) + \alpha \cdot \text{Levy}(\beta)
$$

where:
- \( \alpha \): step size scaling factor,
- \( \text{Levy}(\beta) \): heavy-tailed Lévy distribution with parameter \( \beta \in (1,3] \),  
  typically defined as:

$$
\text{Levy}(\beta) =
\frac{\Gamma(1+\beta) \sin(\pi\beta / 2)}
{\Gamma\left(\frac{1+\beta}{2}\right)\beta 2^{(\beta-1)/2}}
\cdot \frac{u}{|v|^{1/\beta}}
$$



This mechanism enables large exploratory jumps that help escape local minima.



### 5. Stagnation Detection and Escape

TOA monitors fitness improvement using a **stagnation counter** \( S_t \):

$$
S_t =
\begin{cases}
S_{t-1} + 1, & |F_{\text{best}}(t) - F_{\text{best}}(t-1)| < \varepsilon \\
0, & \text{otherwise}
\end{cases}
$$

If \( S_t > S_{\max} \), the algorithm **reinitializes** some of the weakest agents:

$$
\mathbf{x}_i = \text{rand}(L_b, U_b)
\quad \text{for } i \in W
$$

where ( W ) represents the subset of low-performance agents.  
This refresh restores diversity after stagnation.


### 6. Position Update Summary

By combining these dynamic mechanisms, the general agent update rule becomes:

$$
\mathbf{x}_i(t+1) =
\mathbf{x}_i(t)
+ \alpha_1 \cdot \mathcal{E}_1
+ \alpha_2 \cdot \mathcal{E}_2
+ \alpha_3 \cdot \mathcal{E}_3
$$




## Key Characteristics

- **Exploration–Exploitation Balance:** adaptive transition guided by hunger and iteration count.  
- **Elite Coordination:** accelerates convergence through guided communication.  
- **Lévy-Based Jumps:** preserve global diversity and exploration.  
- **Stagnation Escape:** reintroduces randomness when stuck in local minima.  
- **Cryptobiosis Phase:** precision refining near the global optimum.



# multiscale_examples

Solving toy example
```math
\begin{gather}
 - (a^\varepsilon(x) u'(x))' = 0 \ , \quad x \in (0,1) \\
 u(0) = 1 - \varepsilon \ , \quad u(1) = - \varepsilon \cos\left(\frac{1}{\varepsilon}\right) + 3 \\
a(y) = \frac{1}{\sin(y)+2} \ , \quad a^\varepsilon(x) = a\left(\frac{x}{\varepsilon}\right)
\end{gather}
```
for small $\varepsilon$ with different methods, namely:
* Finite Elements
* Generalized finite elements
* Residual free bubbles

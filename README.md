What it is?
---
Codes for [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) using Python 3.5.2

Why it?
---
+ Practicing Python
+ Grasping Deep Learning

How?
---
Keep it simple and STUPID
<br><br>

 :full_moon_with_face:
 :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face: :new_moon_with_face:


 Backpropagation Equations
---
:fire: error $\delta_j^l$ of neuron $j$ in layer $l$
$$
\delta_j^l\equiv \dfrac{\partial C}{\partial z^l_j}
$$

:heavy_check_mark: error $\delta_j^L$ of neuron $j$ in the output layer $L $ <br>
$$
\delta_j^l = \dfrac{\partial C}{\partial z^L_j} \ \sigma'(z^L_j)\tag{BP1}
$$

:heavy_check_mark: error $\delta^l$ in terms of the error in the next layer,  $\ \delta^{l+1}\\ $
$$
\delta^l = (w^{l+1})^T\delta^{l+1})\ \odot\ \sigma'(z^l)\tag{BP2}
$$

:white_check_mark: **An equation for the rate of change of the cost with respect to any bias in the network**<br>
$$
\dfrac{\partial C}{\partial b^l_j}= \delta_j^l \tag{BP3}
$$


:white_check_mark: **An equation for the rate of change of the cost with respect to any weight in the network**<br>
$$
\dfrac{\partial C}{w_{jk}^l}= a^{l-1}_k \delta^l_j \tag{BP4}
$$

import numpy as np
import pandas as pd

x  =  np.array([-3.18518622,  1.74645156,  4.09636192, -2.09368588,  0.44610755,
       -3.98466071,  2.16960775, -4.6211409 , -0.73887275,  4.46604947,
       -3.00068391,  2.70825331, -1.02376892,  0.65867202, -4.79664901,
        4.35680149,  0.51000849,  0.39525456, -0.08423315, -1.13446538])

y = np.array([-3.2955604 , -1.59089425, -0.3995805 , -3.44415531, -1.85278474,
       -4.98282339, -1.93126611, -4.31494598, -1.48874351, -0.80045999,
       -2.95927634, -1.82392489, -2.09714382, -2.03732532, -4.82875516,
       -0.0902187 , -2.88278428, -1.52174129, -2.13553524, -3.14660371])

def gradient_descent(a,b):
    m_curr = b_curr = 0
    iterations = 50000
    n = len(x)
    learning_rate = 0.01
    b = []
    c = []

    for i in range(iterations):
        y_predicted = (m_curr*x) + b_curr
        cost = (1/n)*sum([val**2 for val in (y - y_predicted)])
        pdm = -(2/n)*sum(x*(y - y_predicted))
        pdb = -(2/n)*sum((y-y_predicted))
        b.append(cost)
        c.append([m_curr,b_curr])
        m_curr = m_curr - (learning_rate* pdm)
        b_curr = b_curr - (learning_rate * pdb)
        print(cost)


    d = b.index(min(b))
    print("The best fit line is: ")
    print(f"y = ({c[d][0]})x + ({c[d][1]})")




gradient_descent(x,y)

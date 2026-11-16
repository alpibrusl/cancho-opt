"""The independent answers the examples are held to, in one place so the gate
(scripts/check_examples.py) and the stories (scripts/gen_stories.py) cannot differ:
Newton's method on the logistic loss, the fixed list of starts for the XOR network
and the test that a start solved it."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logistic  # noqa: E402
import mlp_xor  # noqa: E402

XOR_STARTS = ["0.5,-0.4,0.3,0.8,0.1,-0.1,0.7,-0.6,0", "1,1,-1,-1,0.5,-0.5,1,-1,0", "2,-2,2,-2,-1,1,2,2,-1",
              "0.3,0.2,-0.3,0.4,0.1,0.2,0.5,0.5,0.1", "-1,1,1,-1,0,0,1,1,0", "3,3,-3,-3,-1,5,3,-3,0",
              "0.9,-0.8,0.7,-0.6,0.5,-0.4,0.3,-0.2,0.1", "1.5,0.5,-0.5,1.5,0.2,-0.2,1,1,-1"]


def newton_logistic():
    """The independent solve: Newton's method on the same loss, the Hessian written out."""
    w = [0.0, 0.0, 0.0]
    for _ in range(50):
        _, g = logistic.loss_grad(*w)
        h = [[0.0] * 3 for _ in range(3)]
        for x1, x2, _y in logistic.DATA:
            p = 1 / (1 + math.exp(-(w[0] * x1 + w[1] * x2 + w[2])))
            for i, a in enumerate((x1, x2, 1.0)):
                for j, c in enumerate((x1, x2, 1.0)):
                    h[i][j] += p * (1 - p) * a * c / len(logistic.DATA)
        h[0][0] += logistic.L2
        h[1][1] += logistic.L2
        m = [row + [g[i]] for i, row in enumerate(h)]
        for i in range(3):
            piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
            m[i], m[piv] = m[piv], m[i]
            for r in range(3):
                if r != i:
                    f = m[r][i] / m[i][i]
                    m[r] = [a - f * b for a, b in zip(m[r], m[i])]
        w = [w[i] - m[i][3] / m[i][i] for i in range(3)]
    return logistic.loss_grad(*w)[0]


def xor_solved(result):
    """A BFGS result for mlp_xor.py: the loss is near zero and all four points are classified."""
    right = all((mlp_xor.forward(result["x"], a, b)[1] > 0.5) == bool(y) for a, b, y in mlp_xor.DATA)
    return result["fun"] < 1e-3 and right

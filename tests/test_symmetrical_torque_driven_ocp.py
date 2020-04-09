"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import numpy as np

# Load eocarSym
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "eocarSym", str(PROJECT_FOLDER) + "/examples/symmetrical_torque_driven_ocp/eocarSym.py"
)
eocarSym = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eocarSym)


def test_eocarSym():
    ocp = eocarSym.prepare_ocp(biorbd_model_path=str(PROJECT_FOLDER)
        + "/examples/symmetrical_torque_driven_ocp/eocarSym.bioMod")
    sol = ocp.solve()
    )
    sol = nlp.solve()

    q = []
    q_dot = []
    u = []
    for idx in range(nlp.model.nbQ()):
        q.append(np.array(sol["x"][0 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]))
        q_dot.append(
            np.array(sol["x"][1 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()])
        )
        u.append(np.array(sol["x"][2 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]))

    np.testing.assert_almost_equal(q[0][:, 0], np.array((1, 2, 3, 4, 5, 6, 7, 8, 9)))

    # initial and final position
    print(q[0][0, 0])
    print(q[0][-1, 0])
    print(q[1][0, 0])
    print(q[1][-1, 0])
    print(q[2][0, 0])
    print(q[2][-1, 0])
    print(q[3][0, 0])
    print(q[3][-1, 0])
    print(q[4][0, 0])
    print(q[4][-1, 0])
    print(q[5][0, 0])
    print(q[5][-1, 0])
    print(q[6][0, 0])
    print(q[6][-1, 0])
    print(q[7][0, 0])
    print(q[7][-1, 0])
    print(q[8][0, 0])
    print(q[8][-1, 0])
    print(q[9][0, 0])
    print(q[9][-1, 0])
    print(q[10][0, 0])
    print(q[10][-1, 0])
    print(q[11][0, 0])
    print(q[12][-1, 0])
    print(q[12][0, 0])
    print(q[12][-1, 0])

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 114.77652947720107)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    q = []
    q_dot = []
    u = []
    for idx in range(nlp.model.nbQ()):
        q.append(np.array(sol["x"][0 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]))
        q_dot.append(
            np.array(sol["x"][1 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()])
        )
        u.append(np.array(sol["x"][2 * nlp.model.nbQ() + idx :: 3 * nlp.model.nbQ()]))
    # initial and final position
    np.testing.assert_almost_equal(q[0][0, 0], 0)
    np.testing.assert_almost_equal(q[0][-1, 0], -0.8831848069141922)
    np.testing.assert_almost_equal(q[1][0, 0], 0)
    np.testing.assert_almost_equal(q[1][-1, 0], 0.5086923695214512)
    np.testing.assert_almost_equal(q[2][0, 0], -0.5336)
    np.testing.assert_almost_equal(q[2][-1, 0], -0.23114696706313925)
    # initial and final velocities
    np.testing.assert_almost_equal(q_dot[0][0, 0], 0)
    np.testing.assert_almost_equal(q_dot[0][-1, 0], -0.07645884632825446)
    np.testing.assert_almost_equal(q_dot[1][0, 0], 1.4)
    np.testing.assert_almost_equal(q_dot[1][-1, 0], -0.835052743838774)
    np.testing.assert_almost_equal(q_dot[2][0, 0], 0.8)
    np.testing.assert_almost_equal(q_dot[2][-1, 0], 0)
    # initial and final controls
    np.testing.assert_almost_equal(u[0][0, 0], 1.4516128810214546)
    np.testing.assert_almost_equal(u[0][-1, 0], -1.4516128810214546)
    np.testing.assert_almost_equal(u[1][0, 0], 9.81)
    np.testing.assert_almost_equal(u[1][-1, 0], 9.81)
    np.testing.assert_almost_equal(u[2][0, 0], 2.2790322540381487)
    np.testing.assert_almost_equal(u[2][-1, 0], -2.2790322540381487)

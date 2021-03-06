from scipy.integrate import solve_ivp
import numpy as np
import biorbd
from casadi import MX, Function
from matplotlib import pyplot as plt

from biorbd_optim import (
    OptimalControlProgram,
    BidirectionalMapping,
    Mapping,
    Dynamics,
    ProblemType,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
)


def generate_data(biorbd_model, final_time, nb_shooting):
    # Aliases
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_mus = biorbd_model.nbMuscleTotal()
    nb_markers = biorbd_model.nbMarkers()
    dt = final_time / nb_shooting

    # Casadi related stuff
    symbolic_states = MX.sym("x", nb_q + nb_qdot + nb_mus, 1)
    symbolic_controls = MX.sym("u", nb_tau + nb_mus, 1)
    nlp = {
        "model": biorbd_model,
        "nbTau": nb_tau,
        "nbQ": nb_q,
        "nbQdot": nb_qdot,
        "nbMuscle": nb_mus,
        "q_mapping": BidirectionalMapping(Mapping(range(nb_q)), Mapping(range(nb_q))),
        "q_dot_mapping": BidirectionalMapping(Mapping(range(nb_qdot)), Mapping(range(nb_qdot))),
        "tau_mapping": BidirectionalMapping(Mapping(range(nb_tau)), Mapping(range(nb_tau))),
    }
    markers_func = []
    for i in range(nb_markers):
        markers_func.append(
            Function(
                "ForwardKin",
                [symbolic_states],
                [biorbd_model.marker(symbolic_states[:nb_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    dynamics_func = Function(
        "ForwardDyn",
        [symbolic_states, symbolic_controls],
        [Dynamics.forward_dynamics_muscle_excitations_and_torque_driven(symbolic_states, symbolic_controls, nlp)],
        ["x", "u"],
        ["xdot"],
    ).expand()

    def dyn_interface(t, x, u):
        u = np.concatenate([np.zeros(2), u])
        return np.array(dynamics_func(x, u)).squeeze()

    # Generate some muscle excitations
    U = np.random.rand(nb_shooting, nb_mus)

    # Integrate and collect the position of the markers accordingly
    X = np.ndarray((biorbd_model.nbQ() + biorbd_model.nbQdot() + nb_mus, nb_shooting + 1))
    markers = np.ndarray((3, biorbd_model.nbMarkers(), nb_shooting + 1))

    def add_to_data(i, q):
        X[:, i] = q
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(q)).squeeze()

    x_init = np.array([0] * nb_q + [0] * nb_qdot + [0.5] * nb_mus)
    add_to_data(0, x_init)
    for i, u in enumerate(U):
        sol = solve_ivp(dyn_interface, (0, dt), x_init, method="RK45", args=(u,))
        x_init = sol["y"][:, -1]
        add_to_data(i + 1, x_init)

    time_interp = np.linspace(0, final_time, nb_shooting + 1)
    return time_interp, markers, X, U


def prepare_ocp(
    biorbd_model,
    final_time,
    nb_shooting,
    markers_ref,
    excitations_ref,
    q_ref,
    kin_data_to_track="markers",
    show_online_optim=False,
):
    # Problem parameters
    torque_min, torque_max, torque_init = -100, 100, 0
    activation_min, activation_max, activation_init = 0, 1, 0.5
    excitation_min, excitation_max, excitation_init = 0, 1, 0.5

    # Add objective functions
    objective_functions = [
        {"type": Objective.Lagrange.TRACK_MUSCLES_CONTROL, "weight": 1, "data_to_track": excitations_ref},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
    ]
    if kin_data_to_track == "markers":
        objective_functions.append(
            {"type": Objective.Lagrange.TRACK_MARKERS, "weight": 100, "data_to_track": markers_ref},
        )
    elif kin_data_to_track == "q":
        objective_functions.append(
            {
                "type": Objective.Lagrange.TRACK_STATE,
                "weight": 100,
                "data_to_track": q_ref,
                "states_idx": range(biorbd_model.nbQ()),
            },
        )
    else:
        raise RuntimeError("Wrong choice of kin_data_to_track")

    # Dynamics
    variable_type = ProblemType.muscle_excitations_and_torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)
    # Due to unpredictable movement of the forward dynamics that generated the movement, the bound must be larger
    X_bounds.first_node_min[0] = X_bounds.min[0] = X_bounds.last_node_min[0] = -2 * np.pi
    X_bounds.first_node_max[0] = X_bounds.max[0] = X_bounds.last_node_max[0] = 2 * np.pi
    X_bounds.first_node_min[1] = X_bounds.min[1] = X_bounds.last_node_min[1] = -2 * np.pi
    X_bounds.first_node_max[1] = X_bounds.max[1] = X_bounds.last_node_max[1] = 2 * np.pi

    X_bounds.first_node_min += [activation_min] * biorbd_model.nbMuscleTotal()
    X_bounds.first_node_max += [activation_max] * biorbd_model.nbMuscleTotal()
    X_bounds.min += [activation_min] * biorbd_model.nbMuscleTotal()
    X_bounds.max += [activation_max] * biorbd_model.nbMuscleTotal()
    X_bounds.last_node_min += [activation_min] * biorbd_model.nbMuscleTotal()
    X_bounds.last_node_max += [activation_max] * biorbd_model.nbMuscleTotal()

    # Initial guess
    X_init = InitialConditions([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()) + [activation_init] * biorbd_model.nbMuscleTotal())

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [excitation_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [excitation_max] * biorbd_model.nbMuscleTotal(),
    )
    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [excitation_init] * biorbd_model.nbMuscleTotal()
    )

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        nb_shooting,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    # Define the problem
    biorbd_model = biorbd.Model("arm26.bioMod")
    final_time = 2
    n_shooting_points = 29

    # Generate random data to fit
    t, markers_ref, x_ref, muscle_excitations_ref = generate_data(biorbd_model, final_time, n_shooting_points)

    # Track these data
    biorbd_model = biorbd.Model("arm26.bioMod")  # To allow for non free variable, the model must be reloaded
    ocp = prepare_ocp(
        biorbd_model,
        final_time,
        n_shooting_points,
        markers_ref,
        muscle_excitations_ref,
        x_ref[: biorbd_model.nbQ(), :].T,
        show_online_optim=False,
        kin_data_to_track="markers",
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show the results --- #
    muscle_excitations_ref = np.append(muscle_excitations_ref, muscle_excitations_ref[-1:, :], axis=0)

    q, qdot, tau, mus = ProblemType.get_data_from_V(ocp, sol["x"])
    n_q = ocp.nlp[0]["model"].nbQ()
    n_mark = ocp.nlp[0]["model"].nbMarkers()
    n_frames = q.shape[1]
    mus_act = mus[0]
    mus_exci = np.array(mus[1])

    markers = np.ndarray((3, n_mark, q.shape[1]))
    markers_func = []
    for i in range(n_mark):
        markers_func.append(
            Function(
                "ForwardKin",
                [ocp.symbolic_states],
                [biorbd_model.marker(ocp.symbolic_states[:n_q], i).to_mx()],
                ["q"],
                ["marker_" + str(i)],
            ).expand()
        )
    for i in range(n_frames):
        for j, mark_func in enumerate(markers_func):
            markers[:, j, i] = np.array(mark_func(np.append(np.append(q[:, i], qdot[:, i]), mus_act[0][:, i]))).squeeze()

    plt.figure("Markers")
    for i in range(markers.shape[1]):
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers[:, i, :].T, "r--")
    plt.xlabel('Time')
    plt.ylabel('Markers Position')

    plt.figure("Q")
    plt.plot(np.linspace(0, 2, n_shooting_points + 1), x_ref[:n_q, :].T, "k")
    plt.plot(np.linspace(0, 2, n_shooting_points + 1), q.T, "r--")
    plt.xlabel('Time')
    plt.ylabel('Q values')

    plt.figure("Tau")
    plt.step(np.linspace(0, 2, n_shooting_points + 1), tau.T, where="post")
    plt.xlabel('Time')
    plt.ylabel('Torques values (N.m)')

    plt.figure("Muscle excitations")
    plt.step(np.linspace(0, 2, n_shooting_points + 1), muscle_excitations_ref, "k", where="post")
    plt.step(np.linspace(0, 2, n_shooting_points + 1), mus.T, "r--", where="post")
    plt.step(np.linspace(0, 2, n_shooting_points + 1), mus_act[0].T, "g--", where="post")
    plt.xlabel('Time')
    plt.ylabel('Excitation values')

    # --- Plot --- #
    plt.show()

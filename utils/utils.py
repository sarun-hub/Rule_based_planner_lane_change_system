import math, pygame, os
import numpy as np
import control as ct
from utils.vehicle_utils import load_acc_config


# ================== Files ==============================
def get_unique_filepath(base_dir: str, base_filename: str, extension: str) -> str:
    """
    Generate a unique file path by appending a number if the file already exists.

    :param base_dir: Directory where the file will be saved.
    :param base_filename: Base name of the file (without number or extension).
    :param extension: File extension (e.g., '.gif').
    :return: Unique file path.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # Create the directory if it doesn't exist

    # Construct initial file path
    counter = 1
    filepath = os.path.join(base_dir, f"{base_filename}_{counter}{extension}")

    # Increment counter until a unique file path is found
    while os.path.exists(filepath):
        counter += 1
        filepath = os.path.join(base_dir, f"{base_filename}_{counter}{extension}")

    return filepath


# ================== Discretize matrix ==================


def get_discretize_matrix(A, B, C, D, dt=0.1):
    G_s = ct.ss(A, B, C, D)
    G_z = ct.c2d(G_s, dt, "zoh")
    return G_z.A, G_z.B, G_z.C, G_z.D


def compute_next_state(current_state, control_input, A, B, offset=0, return_tuple=True):
    next_state = A @ np.array(current_state).reshape(-1, 1) + B * control_input + offset
    if return_tuple:
        return tuple(next_state)
    else:
        return next_state


def compute_affine_shift(A_d, offset_d):
    AP = np.eye(3) - np.array(A_d)
    AP_inv = np.linalg.pinv(AP)

    x_p = AP_inv @ offset_d
    return x_p


# ================== State space equation ===============


def convert_state3d_to_state2d(
    state_3d, C=np.array([[1, 0, 0], [0, 1, -1]], dtype=float)
):
    state_2d = C @ state_3d
    return tuple(state_2d)


# Helper TODO 23/12/2025 ⭐
# This assume the system. later make it better
def get_A_and_offset():
    aggressive, h, delta_min = load_acc_config()
    # setup system
    A_u = np.array(
        [[0, 1, -1], [0, 0, 0], [aggressive / h, 1 / h, -(1 + aggressive * h) / h]],
        dtype=float,
    )
    B_u = np.array([[0], [1], [0]])
    C_u = np.array([[1, 0, 0], [0, 1, -1]], dtype=float)
    D_u = np.array([[0], [0]], dtype=float)
    offset = np.array([[0], [0], [-aggressive / h * delta_min]])

    # use augmented matrix to find discretized matrix
    A_aug = np.block([[A_u, offset], [np.zeros((1, 4))]])
    B_aug = np.vstack((B_u, np.zeros((1, 1))))
    C_aug = np.hstack((C_u, np.zeros((2, 1))))
    D_aug = np.zeros((2, 1))

    A_d_aug, B_d_aug, C_d_aug, _ = get_discretize_matrix(A_aug, B_aug, C_aug, D_aug)

    # Use A 3x3 from augmented system
    A_d = A_d_aug[:3, :3]
    B_d = B_d_aug[:3, :]
    C_d = C_d_aug[:, :3]
    offset_d = A_d_aug[:3, 3:4]  # the last column (only 3 rows) is offset

    return A_d, offset_d


A_d, offset_d = get_A_and_offset()
x_p = compute_affine_shift(A_d, offset_d)

# convert actual state 2d to shifted state 3d 
def convert_state2d_to_state3d_pass_affine(
    state_2d, C=np.array([[1, 0, 0], [0, 1, -1]], dtype=float)
):
    y_p = C @ x_p
    T = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
    CT = C @ T
    z_wanted = np.linalg.pinv(CT) @ (state_2d - y_p)
    x_wanted = T @ z_wanted
    return x_wanted

# convert linear (shifted) state 3d to actual (affine) state 3d 
def convert_linear_to_affine_3d(state_3d):
    arr = np.asarray(state_3d).reshape(-1)
    if arr.size == 3:
        state_3d = arr.reshape(3, 1)
    else :
        raise ValueError(f"Expect 3 elements, now it's {arr.size}.")
    return state_3d + x_p

# convert actual (affine) state 3d to linear (shifted) state 3d 
def convert_affine_to_linear_3d(state_3d):
    arr = np.asarray(state_3d).reshape(-1)
    if arr.size == 3:
        state_3d = arr.reshape(3, 1)
    else :
        raise ValueError(f"Expect 3 elements, now it's {arr.size}.")
    return state_3d - x_p

# convert actual (affine) state 2d to linear (shifted) state 2d 
def convert_affine_to_linear_2d(state_2d):
    arr = np.asarray(state_2d).reshape(-1)
    if arr.size == 2:
        state_2d = arr.reshape(2, 1)
    else :
        raise ValueError(f"Expect 2 elements, now it's {arr.size}.")
    C=np.array([[1, 0, 0], [0, 1, -1]])
    y_p = C @ x_p

    return state_2d - y_p


if __name__ == "__main__":
    print("x_p")
    print(compute_affine_shift(A_d, offset_d))

    print("y_p")
    print( np.array([[1, 0, 0], [0, 1, -1]]) @ compute_affine_shift(A_d, offset_d))

    y_initial = np.array([[10], [0]], dtype=float)
    print("3d y shifted")
    print(convert_state2d_to_state3d_pass_affine(y_initial))

    print("linear 2d")
    print(convert_affine_to_linear_2d(y_initial))

    test_x3d = (1, 2, 3)
    print("x3d")
    print(convert_linear_to_affine_3d(test_x3d))

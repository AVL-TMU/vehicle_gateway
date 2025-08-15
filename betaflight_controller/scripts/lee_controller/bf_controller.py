#!/usr/bin/env python3
import numpy as np
from numpy.linalg import norm
import math

def calculate_controller_output(position_W, r_position_W,
                                velocity_W, r_velocity_W,
                                R_B_W, uav_mass, gravity,
                                r_acceleration_W,
                                position_gain, velocity_gain,
                                r_yaw, angular_velocity_B,
                                inertia_matrix,
                                attitude_gain, angular_rate_gain,
                                r_yaw_rate):
    """
    Geometric controller (ENU world, FLU body).
    Returns:
      - controller_torque_thrust: [tau_x, tau_y, tau_z, thrust_N]
      - R_d_w: desired rotation matrix (world<-body)
      - debug: dict with e_p, e_v, I_a_d, etc (for plotting)
    """
    # Position/velocity errors
    e_p = position_W - r_position_W
    e_v = velocity_W - r_velocity_W

    # Desired specific force * mass
    I_a_d = -position_gain * e_p \
            - velocity_gain * e_v \
            + uav_mass * gravity * np.array([0.0, 0.0, 1.0]) \
            + uav_mass * r_acceleration_W

    # Thrust along current body z axis
    thrust = float(np.dot(I_a_d, R_B_W[:, 2]))

    # Desired body axes in world (ENU)
    B_z_d = I_a_d / (norm(I_a_d) + 1e-9)
    B_x_d = np.array([math.cos(r_yaw), math.sin(r_yaw), 0.0])
    B_y_d = np.cross(B_z_d, B_x_d)
    B_y_d = B_y_d / (norm(B_y_d) + 1e-9)
    B_x_d = np.cross(B_y_d, B_z_d)

    # Desired rotation (world <- body)
    R_d_w = np.column_stack([B_x_d, B_y_d, B_z_d])

    # Attitude error / rate error (not used by Betaflight, but we keep it for debugging)
    e_R_matrix = 0.5 * (R_d_w.T @ R_B_W - R_B_W.T @ R_d_w)
    e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])

    omega_ref = r_yaw_rate * np.array([0.0, 0.0, 1.0])
    e_omega = angular_velocity_B - R_B_W.T @ R_d_w @ omega_ref
    tau = -attitude_gain * e_R - angular_rate_gain * e_omega \
          + np.cross(angular_velocity_B, inertia_matrix @ angular_velocity_B)

    dbg = {
        "e_p": e_p, "e_v": e_v,
        "I_a_d": I_a_d, "thrust": thrust,
        "e_R": e_R, "e_omega": e_omega
    }
    return np.concatenate([tau, [thrust]]), R_d_w, dbg


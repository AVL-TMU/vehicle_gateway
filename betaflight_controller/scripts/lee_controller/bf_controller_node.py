#!/usr/bin/env python3
import math, time, numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Float64
from bf_controller import calculate_controller_output

def clamp(x, lo, hi): return max(lo, min(hi, x))
def wrap_pi(a):
    while a >  math.pi: a -= 2.0*math.pi
    while a < -math.pi: a += 2.0*math.pi
    return a

def quat_to_R_wb(x, y, z, w):
    # Quaternion (x,y,z,w) -> rotation (world <- body), ENU/FLU
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),   2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),   1 - 2*(xx+yy)]
    ])

class BFControllerNode(Node):
    """
    Betaflight geometric controller node (two-file split like PX4 version).
    - Subscribes: /odometry (nav_msgs/Odometry, ENU/FLU), /command/pose (PoseStamped)
    - Publishes:  /joy (sensor_msgs/Joy)   -> your MSP bridge sends MSP_SET_RAW_RC to Betaflight
                  /bf_ctrl/debug/* (Vector3/Float64) for easy plotting/debugging
    Frames:
      World W = ENU (+Z up). Body B = FLU.
      Betaflight RC is FRD; we flip roll/pitch stick signs accordingly.
    """
    def __init__(self):
        super().__init__('bf_controller_node')

        # ==== Physical & controller params (copied from your PX4 node) ====
        self.uav_mass = 1.5
        self.gravity  = 9.81
        
        
        '''self.position_gain = 0.5*np.array([7.0, 7.0, 6.0])
        self.velocity_gain = 0.25*np.array([6.0, 6.0, 3.0])
        self.attitude_gain = 0.25*np.array([3.5, 3.5, 0.3])        
        self.angular_rate_gain = 0.5*np.array([0.5, 0.5, 0.1])'''



        self.position_gain = np.array([0.5, 0.5, 1.0])  # Reduced from [7,7,6]
        self.velocity_gain = np.array([0.4, 0.4, 0.3])  # Reduced from [6,6,3]
        self.attitude_gain = np.array([0.3, 0.3, 0.1])  # Reduced from [3.5,3.5,0.3]
        self.angular_rate_gain = np.array([0.1, 0.1, 0.05])  # Reduced from [0.5,0.5,0.1]

        
        
        self.inertia_matrix = np.array([[0.029125, 0, 0],
                                        [0, 0.029125, 0],
                                        [0, 0, 0.055225]])

        # ==== Trajectory ====
        self.mode = 'circle'   # 'hover' or 'circle'
        self.z_ref = 1.0
        self.center = np.array([0.0, 0.0])
        self.radius = 1.0
        self.period = 12.0
        self.yaw_mode = 'fixed'   # 'fixed' or 'tangent'
        self.r_yaw_fixed = 0.0
        self.r_yaw_rate_ff = 0.0

        # ==== Betaflight /joy mapping ====
        # Set these to match your Receiver tab (use your axis probe script if unsure)
        self.AXIS = {'yaw': 0, 'throttle': 1, 'roll': 2, 'pitch': 3}
        self.max_tilt_deg = 30.0
        self.hover_thr_stick = 0.55
        self.thr_gain = 0.16           # stick per (m/s^2): az = T/m - g
        self.yaw_k = 0.4               # yaw error -> stick
        self.yaw_rate_to_stick = 0.2   # rad/s -> stick (FF)

        # FLU -> FRD stick sign flips (right/forward should be +stick in BF)
        self.sgn_roll  = -1.0
        self.sgn_pitch = -1.0

        # ==== Smoothing & timing ====
        self.rate_hz = 50.0
        self.stick_tau = 0.5
        self.slew_rp = 0.25
        self.slew_yaw = 0.25
        self.slew_thr = 0.6

        self.arm_delay = 2.0
        self.arm_phase = 2.0
        self.ramp_after_arm = 1.5

        # ==== State ====
        self.posW = np.zeros(3)
        self.velW = np.zeros(3)
        self.Rwb  = np.eye(3)
        self.yawW = 0.0
        self.omegaB = np.zeros(3)     
        self.have_odom = False

        self.r_position_W = np.array([0.0, 0.0, self.z_ref])
        self.r_velocity_W = np.zeros(3)
        self.r_acceleration_W = np.zeros(3)
        self.r_yaw = self.r_yaw_fixed
        self.r_yaw_rate = 0.0

        self.roll_s = 0.0; self.pitch_s = 0.0; self.yaw_s = 0.0; self.thr_s = -1.0
        self.t0 = time.time(); self.last_t = self.t0

        # ==== IO ====
        self.sub_odom = self.create_subscription(
            Odometry, '/model/iris_with_Betaflight/model/iris_with_standoffs/odometry',
            self.on_odom, 10)
            
        self.sub_imu = self.create_subscription(
            Imu, '/model/iris_with_Betaflight/model/iris_with_standoffs/imu', 
            self.on_imu, 10)
          
        self.sub_cmd = self.create_subscription(
            PoseStamped, '/command/pose', self.on_command_pose, 10)
        self.pub_joy = self.create_publisher(Joy, '/joy', 10)

        # Debug publishers (for rqt_plot, ros2 topic echo)
        self.pub_err_p = self.create_publisher(Vector3, '/bf_ctrl/debug/err_pos', 10)
        self.pub_err_v = self.create_publisher(Vector3, '/bf_ctrl/debug/err_vel', 10)
        self.pub_ang_des = self.create_publisher(Vector3, '/bf_ctrl/debug/angles_des_deg', 10)
        self.pub_thrustN = self.create_publisher(Float64, '/bf_ctrl/debug/thrust_N', 10)
        self.pub_thrstick = self.create_publisher(Float64, '/bf_ctrl/debug/throttle_stick', 10)

        self.timer = self.create_timer(1.0/self.rate_hz, self.tick)

    # ----- Callbacks -----
    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        self.posW = np.array([p.x, p.y, p.z], dtype=float)
        self.velW = np.array([v.x, v.y, v.z], dtype=float)     # ENU
        self.Rwb  = quat_to_R_wb(q.x, q.y, q.z, q.w)           # ENU/FLU
        self.yawW = math.atan2(self.Rwb[1,0], self.Rwb[0,0])   # heading
        self.have_odom = True


    def on_imu(self, msg: Imu):
        self.omegaB = np.array([
            msg.angular_velocity.x,
            -msg.angular_velocity.y,
            -msg.angular_velocity.z
        ], dtype=float)


    def on_command_pose(self, msg: PoseStamped):
        # Reference pose (position + yaw from quaternion)
        self.r_position_W = np.array([msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z], dtype=float)
        # Yaw from quaternion (ENU)
        qx,qy,qz,qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        Rref = quat_to_R_wb(qx,qy,qz,qw)
        self.r_yaw = math.atan2(Rref[1,0], Rref[0,0])
        self.r_velocity_W[:] = 0.0
        self.r_acceleration_W[:] = 0.0
        self.r_yaw_rate = 0.0

    # ----- Reference generation -----
    def update_reference(self, t):
        if self.mode == 'hover':
            self.r_position_W = np.array([0.5, 0.5, self.z_ref])
            self.r_velocity_W[:] = 0.0
            self.r_acceleration_W[:] = 0.0
            self.r_yaw = self.r_yaw_fixed
            self.r_yaw_rate = 0.0
            return

        # circle
        w = 2.0*math.pi/self.period
        s = w*(t - self.t0)
        xr = self.center[0] + self.radius*math.cos(s)
        yr = self.center[1] + self.radius*math.sin(s)
        zr = self.z_ref
        vxr = -self.radius*w*math.sin(s)
        vyr =  self.radius*w*math.cos(s)
        axr = -self.radius*w*w*math.cos(s)
        ayr = -self.radius*w*w*math.sin(s)

        self.r_position_W = np.array([xr, yr, zr])
        self.r_velocity_W = np.array([vxr, vyr, 0.0])
        self.r_acceleration_W = np.array([axr, ayr, 0.0])

        if self.yaw_mode == 'tangent':
            self.r_yaw = math.atan2(vyr, vxr)
            self.r_yaw_rate = w
        else:
            self.r_yaw = self.r_yaw_fixed
            self.r_yaw_rate = self.r_yaw_rate_ff

    # ----- Main control loop -----
    def tick(self):
        t = time.time()
        dt = max(t - self.last_t, 1.0/self.rate_hz)
        self.last_t = t

        joy = Joy(); joy.header.stamp = self.get_clock().now().to_msg()
        arm = 0
        yaw_cmd, roll_cmd, pitch_cmd, thr_cmd = self.yaw_s, self.roll_s, self.pitch_s, -1.0

        if not self.have_odom or (t - self.t0) < self.arm_delay:
            pass
        elif (t - self.t0) < self.arm_delay + self.arm_phase:
            arm = 1
        else:
            arm = 1
            arm_t = (t - self.t0) - (self.arm_delay + self.arm_phase)
            if arm_t < self.ramp_after_arm:
                k = arm_t / self.ramp_after_arm
                thr_cmd = -1.0 + (self.hover_thr_stick + 1.0) * k
            else:
                # pick reference
                self.update_reference(t)

                # === Geometric controller (math-only file) ===
                ctrl, Rdes, dbg = calculate_controller_output(
                    position_W=self.posW, r_position_W=self.r_position_W,
                    velocity_W=self.velW, r_velocity_W=self.r_velocity_W,
                    R_B_W=self.Rwb, uav_mass=self.uav_mass, gravity=self.gravity,
                    r_acceleration_W=self.r_acceleration_W,
                    position_gain=self.position_gain, velocity_gain=self.velocity_gain,
                    r_yaw=self.r_yaw, angular_velocity_B=self.omegaB,
                    inertia_matrix=self.inertia_matrix,
                    attitude_gain=self.attitude_gain, angular_rate_gain=self.angular_rate_gain,
                    r_yaw_rate=self.r_yaw_rate
                )
                thrust_N = float(ctrl[3])

                # Desired angles from Rdes (ZYX)
                max_tilt = math.radians(self.max_tilt_deg)
                theta_des = clamp(-math.asin(clamp(-Rdes[2,0], -1.0, 1.0)), -max_tilt, max_tilt)  # pitch (FLU)
                phi_des   = clamp( math.atan2(Rdes[2,1], Rdes[2,2]),        -max_tilt, max_tilt)  # roll  (FLU)
                
                #print("theta_des", theta_des, "phi_des",phi_des)

                # Map to Betaflight ANGLE sticks with FLU->FRD sign
                pitch_des = clamp(self.sgn_pitch * (theta_des / max_tilt), -1.0, 1.0)
                roll_des  = clamp(self.sgn_roll  * (phi_des   / max_tilt), -1.0, 1.0)

                # Thrust -> throttle stick: az = T/m - g
                az = (thrust_N / self.uav_mass) - self.gravity
                thr_des = clamp(self.hover_thr_stick + self.thr_gain * az, -1.0, 1.0)

                # Yaw
                yaw_err = wrap_pi(self.r_yaw - self.yawW)
                yaw_des = clamp(self.yaw_rate_to_stick * self.r_yaw_rate + self.yaw_k * yaw_err, -0.5, 0.5)

                # LPF + slew for smooth RC
                a = math.exp(-dt / self.stick_tau) if self.stick_tau > 1e-3 else 0.0
                roll_pf  = a*self.roll_s  + (1.0-a)*roll_des
                pitch_pf = a*self.pitch_s + (1.0-a)*pitch_des
                yaw_pf   = a*self.yaw_s   + (1.0-a)*yaw_des
                thr_pf   = a*self.thr_s   + (1.0-a)*thr_des

                def slew(prev, target, rate): return prev + clamp(target - prev, -rate*dt, rate*dt)
                roll_cmd  = slew(self.roll_s,  roll_pf,  self.slew_rp)
                pitch_cmd = slew(self.pitch_s, pitch_pf, self.slew_rp)
                yaw_cmd   = slew(self.yaw_s,   yaw_pf,   self.slew_yaw)
                thr_cmd   = slew(self.thr_s,   thr_pf,   self.slew_thr)

                self.roll_s, self.pitch_s, self.yaw_s, self.thr_s = roll_cmd, pitch_cmd, yaw_cmd, thr_cmd

                # ---- Debug topics ----
                ep = dbg["e_p"]; ev = dbg["e_v"]
                msg_ep = Vector3(x=float(ep[0]), y=float(ep[1]), z=float(ep[2]))
                msg_ev = Vector3(x=float(ev[0]), y=float(ev[1]), z=float(ev[2]))
                msg_angles = Vector3(
                    x=math.degrees(phi_des),   # roll [deg]
                    y=math.degrees(theta_des), # pitch [deg]
                    z=math.degrees(self.r_yaw) # desired yaw [deg] (for visibility)
                )
                self.pub_err_p.publish(msg_ep)
                self.pub_err_v.publish(msg_ev)
                self.pub_ang_des.publish(msg_angles)
                self.pub_thrustN.publish(Float64(data=thrust_N))
                self.pub_thrstick.publish(Float64(data=self.thr_s))

        # Publish /joy with your axis mapping
        axes = [0.0, 0.0, 0.0, 0.0]
        axes[self.AXIS['yaw']]      = self.yaw_s
        axes[self.AXIS['throttle']] = self.thr_s
        axes[self.AXIS['roll']]     = self.roll_s
        axes[self.AXIS['pitch']]    = self.pitch_s
        joy.axes = axes
        joy.buttons = [arm, 0, 0, 0]
        self.pub_joy.publish(joy)

def main():
    rclpy.init()
    node = BFControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # disarm ramp
        for _ in range(20):
            j = Joy(); j.axes = [0.0, -1.0, 0.0, 0.0]; j.buttons = [0,0,0,0]
            node.pub_joy.publish(j); time.sleep(0.05)
        rclpy.shutdown()

if __name__ == '__main__':
    main()


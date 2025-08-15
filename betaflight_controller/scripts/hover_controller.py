#!/usr/bin/env python3
import time, math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy

def clamp(x, lo, hi): return max(lo, min(hi, x))

class BFHoverZ(Node):
    def __init__(self):
        super().__init__('bf_hover_z')
        # ====== Params you may tweak ======
        self.z_ref = 1.0               # target altitude [m]
        self.hover_thr_stick = 0.55    # your working hover stick ([-1,1])
        self.kp = 0.60                 # stick per meter error
        self.kd = 0.35                 # stick per (m/s)
        self.ki = 0.10                 # stick per (m*s)
        self.lpf_tau = 0.20            # throttle smoothing (s), 0.2–0.5 recommended
        self.arm_delay = 2.0           # s at thr=-1 before arming
        self.ramp_after_arm = 1.5      # s to ramp up to hover
        self.rate_hz = 50.0
        # ==================================

        self.alt_int = 0.0
        self.have_odom = False
        self.z = 0.0
        self.vz = 0.0
        self.start = time.time()
        self.thr_filt = -1.0           # filtered throttle stick

        self.sub = self.create_subscription(
            Odometry,
            '/model/iris_with_Betaflight/model/iris_with_standoffs/odometry',
            self.on_odom, 10)
        self.pub = self.create_publisher(Joy, '/joy', 10)
        self.timer = self.create_timer(1.0/self.rate_hz, self.tick)

    def on_odom(self, msg: Odometry):
        self.z = msg.pose.pose.position.z
        self.vz = msg.twist.twist.linear.z
        self.have_odom = True

    def tick(self):
        t = time.time() - self.start
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()

        # Default sticks: level & no yaw
        yaw_stick = 0.0
        roll_stick = 0.0
        pitch_stick = 0.0
        arm_button = 0
        thr_cmd = -1.0

        if not self.have_odom:
            # Wait for odom, keep disarmed
            pass

        elif t < self.arm_delay:
            # pre-arm: throttle min
            pass

        elif t < self.arm_delay + 2.0:
            # arm with throttle min
            arm_button = 1

        else:
            arm_button = 1
            # Optional gentle ramp for the first seconds after arming
            arm_t = t - (self.arm_delay + 2.0)
            if arm_t < self.ramp_after_arm:
                k = arm_t / self.ramp_after_arm
                thr_cmd = -1.0 + (self.hover_thr_stick + 1.0) * k
            else:
                # Simple PID on altitude (stick domain)
                ez  = self.z_ref - self.z
                evz = 0.0 - self.vz
                # Integrator
                self.alt_int += ez * (1.0/self.rate_hz)
                # Raw command
                thr_cmd = (self.hover_thr_stick
                           + self.kp*ez + self.kd*evz + self.ki*self.alt_int)
                thr_cmd = clamp(thr_cmd, -1.0, 1.0)
                # Anti-windup: freeze I when saturated in same direction
                if (thr_cmd == 1.0 and (self.kp*ez + self.kd*evz + self.ki*self.alt_int) > 0) or \
                   (thr_cmd == -1.0 and (self.kp*ez + self.kd*evz + self.ki*self.alt_int) < 0):
                    self.alt_int -= ez * (1.0/self.rate_hz)

        # Smooth throttle to avoid jumps
        dt = 1.0/self.rate_hz
        alpha = math.exp(-dt / self.lpf_tau) if self.lpf_tau > 1e-3 else 0.0
        self.thr_filt = alpha*self.thr_filt + (1.0-alpha)*thr_cmd
        thr_stick = self.thr_filt

        # Your axis order: [yaw, throttle, roll, pitch]
        msg.axes = [yaw_stick, thr_stick, roll_stick, pitch_stick]
        msg.buttons = [arm_button, 0, 0, 0]
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = BFHoverZ()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Disarm cleanly
        for _ in range(20):
            m = Joy(); m.axes = [0.0, -1.0, 0.0, 0.0]; m.buttons = [0,0,0,0]
            node.pub.publish(m); time.sleep(0.05)
        rclpy.shutdown()

if __name__ == '__main__':
    main()


import math
import queue
import signal
import threading
import time
from typing import Optional
from lerobot.utils.robot_utils import precise_sleep

# Sentinel token used in place of a numeric gripper position to trigger torque grasping
GRASP = "GRASP"

# Safe resting / parked pose executed on program exit (Ctrl+C, Q, ESC) before torque-off
HOME_POSE = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -100.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": 75.0,
    "wrist_roll.pos": -90.0,
    "gripper.pos": 2.0,
}

# Active scanning / ready pose held while searching for and sweeping toward objects
STOW_HOLD_POSE = {
    "shoulder_pan.pos": -2.42,
    "shoulder_lift.pos": -51.08,
    "elbow_flex.pos": 21.85,
    "wrist_flex.pos": 96.48,
    "wrist_roll.pos": -85.14,
    "gripper.pos": 1.01,
}

PICK_AND_DROP_SEQUENCE = [
    ({'shoulder_pan.pos': -9.32, 'shoulder_lift.pos': 77.05, 'elbow_flex.pos': 8.66, 'wrist_flex.pos': 2.29, 'wrist_roll.pos': -97.98, 'gripper.pos': 64.5}, 1.2),  # (target_pose, duration_seconds)
    ({'shoulder_pan.pos': -9.58, 'shoulder_lift.pos': 103.25, 'elbow_flex.pos': -20.79, 'wrist_flex.pos': 2.11, 'wrist_roll.pos': -97.8, 'gripper.pos': 64.44}, 1.2),  # (target_pose, duration_seconds)
    ({'shoulder_pan.pos': -9.49, 'shoulder_lift.pos': 103.16, 'elbow_flex.pos': -20.7, 'wrist_flex.pos': 2.2, 'wrist_roll.pos': -97.36, 'gripper.pos': GRASP}, 1.2),  # (target_pose, duration_seconds)
    ({"shoulder_pan.pos": -7.03, "shoulder_lift.pos": -92.26, "elbow_flex.pos": 21.76, "wrist_flex.pos": 68.84, "wrist_roll.pos": -94.46}, 1.2),
]

BIN_WAYPOINTS = [
    # Bin 0 (Left)
    [
        ({"shoulder_pan.pos": -10.37, "shoulder_lift.pos": -64.92, "elbow_flex.pos": -88.66, "wrist_flex.pos": -27.52, "wrist_roll.pos": -94.46}, 1.2),
        ({"gripper.pos": 65.0}, 0.4),
    ],
    # Bin 1 (Rear-Center)
    [
        ({"shoulder_pan.pos": 4.92, "shoulder_lift.pos": -67.12, "elbow_flex.pos": -87.78, "wrist_flex.pos": -27.52, "wrist_roll.pos": -94.46}, 1.2),
        ({"gripper.pos": 65.0}, 0.4),
    ],
]


class _EmptyGraspAbort(Exception):
    """Internal control-flow signal raised when torque grasping closes on empty air."""

    def __init__(self, current_pose: dict[str, float]):
        super().__init__("Empty grasp detected")
        self.current_pose = current_pose


class ArmHoldController:
    def __init__(
        self,
        robot,
        bin_waypoints: list = BIN_WAYPOINTS,
        hold_pose: dict[str, float] = STOW_HOLD_POSE,
        home_pose: dict[str, float] = HOME_POSE,
        fps: int = 30,
        max_speed: float = 50.0,
        lekiwi_mode: bool = False,
        # Torque-grasp tuning parameters
        grasp_torque_limit: int = 140,
        empty_closed_pos: float = 3.0,
        grasp_step_deg: float = 2.0,
        stall_error_deg: float = 4.5,
        load_threshold: float = 120.0,
        confirm_ticks: int = 4,
        # Smooth visual-servo pan sweep parameters
        pan_sign: float = -1.0,
        pan_kp: float = 0.95,
        pan_kd: float = 0.18,
        sweep_max_speed: float = 15.0,
        sweep_max_accel: float = 45.0,
        sweep_max_decel: float = 120.0,
        pan_deadband_deg: float = 2.0,
        angle_zero_tol_deg: float = 2.5,
        pan_limits_deg: tuple[float, float] = (-50.0, 50.0),
        return_to_stow_on_lost: bool = True,
        lost_grace_sec: float = 4.0,
        stow_on_start: bool = True,
    ):
        self.robot = robot
        self.bin_waypoints = bin_waypoints
        self.base_hold_pose = dict(hold_pose)
        self.hold_pose = dict(hold_pose)
        self.home_pose = dict(home_pose)
        self.fps = fps
        self.max_speed = max_speed
        self.lekiwi_mode = lekiwi_mode
        self.prefix = "arm_" if lekiwi_mode else ""

        self.grasp_torque_limit = grasp_torque_limit
        self.empty_closed_pos = empty_closed_pos
        self.grasp_step_deg = grasp_step_deg
        self.stall_error_deg = stall_error_deg
        self.load_threshold = load_threshold
        self.confirm_ticks = confirm_ticks

        self.pan_sign = pan_sign
        self.pan_kp = pan_kp
        self.pan_kd = pan_kd
        self.sweep_max_speed = sweep_max_speed
        self.sweep_max_accel = sweep_max_accel
        self.sweep_max_decel = sweep_max_decel
        self.pan_deadband_deg = pan_deadband_deg
        self.angle_zero_tol_deg = angle_zero_tol_deg
        self.pan_limits_deg = pan_limits_deg
        self.return_to_stow_on_lost = return_to_stow_on_lost
        self.lost_grace_sec = lost_grace_sec
        self.stow_on_start = stow_on_start

        self._pan_lock = threading.Lock()
        self._visual_angle_deg: Optional[float] = None
        self._prev_visual_angle_deg: Optional[float] = None
        self._angle_rate_deg_s: float = 0.0
        self._last_seen_time: Optional[float] = None
        self._force_return_to_stow: bool = False
        self._current_pan_vel: float = 0.0

        self.last_grasp_success: bool = False
        self._cmd_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._busy_event = threading.Event()
        self._thread = threading.Thread(target=self._control_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.5)

    def set_home_pose(self, pose: dict[str, float]) -> None:
        """Updates the target homing pose used during shutdown_to_home()."""
        with self._pan_lock:
            self.home_pose.update({k: float(v) for k, v in pose.items()})

    def shutdown_to_home(self, home_pose: Optional[dict[str, float]] = None, duration: float = 1.5) -> None:
        """
        Safely stops the background control loop and smoothly moves the arm from its
        current physical posture to `home_pose` (defaults to `self.home_pose`) before
        disconnecting. Protected against duplicate Ctrl+C interrupts during travel.
        """
        if home_pose is not None:
            self.set_home_pose(home_pose)

        prev_sigint = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.stop()
            self._set_gripper_torque_limit(1000)

            obs = self.robot.get_observation()
            current = {k: float(obs[f"{self.prefix}{k}"]) for k in self.home_pose.keys()}
            with self._pan_lock:
                target_home = dict(self.home_pose)

            print(f"[ArmHoldController] Moving to HOME_POSE: {target_home}")
            self._run_waypoints([(target_home, duration)], current, pan_override=None, ignore_stop=True)
            time.sleep(0.15)
        except Exception as e:
            print(f"[ArmHoldController] Homing on shutdown encountered an error: {e}")
        finally:
            signal.signal(signal.SIGINT, prev_sigint)

    @property
    def is_busy(self) -> bool:
        return self._busy_event.is_set()

    @property
    def current_pan(self) -> float:
        with self._pan_lock:
            return float(self.hold_pose["shoulder_pan.pos"])

    @property
    def lost_elapsed_sec(self) -> Optional[float]:
        """Seconds elapsed since the last valid target was seen, or None if never seen / currently locked."""
        with self._pan_lock:
            if self._visual_angle_deg is not None or self._last_seen_time is None:
                return None
            return max(0.0, time.perf_counter() - self._last_seen_time)

    @property
    def is_aligned(self) -> bool:
        with self._pan_lock:
            return (
                not self._busy_event.is_set()
                and self._visual_angle_deg is not None
                and abs(self._visual_angle_deg) <= self.angle_zero_tol_deg
                and abs(self._current_pan_vel) <= 1.5
            )

    def update_visual_angle(self, angle_rad: Optional[float]) -> None:
        """Feeds the latest visual target angle (radians), or None if no valid target is present."""
        now = time.perf_counter()
        with self._pan_lock:
            if angle_rad is not None:
                new_deg = math.degrees(angle_rad)
                if self._prev_visual_angle_deg is not None and self._last_seen_time is not None:
                    dt = now - self._last_seen_time
                    if 0.005 <= dt <= 0.25:
                        raw_rate = (new_deg - self._prev_visual_angle_deg) / dt
                        self._angle_rate_deg_s = 0.4 * raw_rate + 0.6 * self._angle_rate_deg_s
                    else:
                        self._angle_rate_deg_s = 0.0
                else:
                    self._angle_rate_deg_s = 0.0

                self._prev_visual_angle_deg = new_deg
                self._visual_angle_deg = new_deg
                self._last_seen_time = now
                self._force_return_to_stow = False
            else:
                self._visual_angle_deg = None
                self._prev_visual_angle_deg = None
                self._angle_rate_deg_s = 0.0

    def reset_to_stow_pan(self, wait: bool = False, timeout_s: float = 3.0) -> None:
        """Immediately begins smoothly returning shoulder_pan back to STOW_HOLD_POSE."""
        with self._pan_lock:
            self._visual_angle_deg = None
            self._prev_visual_angle_deg = None
            self._angle_rate_deg_s = 0.0
            self._force_return_to_stow = True
        if wait and self._thread.is_alive() and not self._stop_event.is_set():
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                if abs(self.current_pan - self.base_hold_pose["shoulder_pan.pos"]) <= 0.3:
                    break
                time.sleep(0.02)

    def send_trajectory(
        self,
        sequence: list[tuple[dict, float]],
        bin_index: int,
        pan_angle: Optional[float] = None,
        return_to_hold: bool = True,
        wait: bool = False,
    ) -> None:
        """
        Executes `sequence` (optionally overriding `shoulder_pan.pos` with `pan_angle`),
        then finishes with `bin_waypoints[bin_index]` using its own native pan angle.
        If any `GRASP` step fails to detect an object, aborts immediately to `hold_pose`.
        """
        if not (0 <= bin_index < len(self.bin_waypoints)):
            raise IndexError(f"bin_index {bin_index} out of range (0..{len(self.bin_waypoints) - 1})")

        done_event = threading.Event()
        self._busy_event.set()
        self._cmd_queue.put((sequence, bin_index, pan_angle, return_to_hold, done_event))
        if wait:
            done_event.wait()

    # ------------------------------------------------------------------
    # Hardware Bus Helpers (for torque limit & load sensing)
    # ------------------------------------------------------------------
    def _set_gripper_torque_limit(self, limit: int) -> None:
        if hasattr(self.robot, "bus"):
            try:
                self.robot.bus.write("Torque_Limit", "gripper", int(limit), normalize=False)
                time.sleep(0.01)
            except Exception as e:
                print(f"[ArmHoldController] Torque_Limit write failed: {e}")

    def _read_gripper_load(self) -> float:
        if hasattr(self.robot, "bus"):
            try:
                val = self.robot.bus.read("Present_Load", "gripper", normalize=False)
                return abs(float(val))
            except Exception:
                pass
        return 0.0

    def _format_action(self, pose: dict[str, float]) -> dict[str, float]:
        action = {f"{self.prefix}{k}": float(v) for k, v in pose.items()}
        if self.lekiwi_mode:
            action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        return action

    # ------------------------------------------------------------------
    # Torque-Controlled Grasping Routine
    # ------------------------------------------------------------------
    def _execute_torque_grasp(self, current: dict[str, float], timeout_s: float = 2.5) -> float:
        self._set_gripper_torque_limit(self.grasp_torque_limit)

        obs = self.robot.get_observation()
        actual_pos = float(obs[f"{self.prefix}gripper.pos"])
        cmd_pos = actual_pos
        prev_pos = actual_pos

        firm_ticks = 0
        self.last_grasp_success = False
        max_steps = max(int(timeout_s * self.fps), 1)

        for _ in range(max_steps):
            if self._stop_event.is_set():
                break
            t0 = time.perf_counter()

            cmd_pos = max(0.0, min(cmd_pos - self.grasp_step_deg, actual_pos - 6.0))
            pose_cmd = {**current, "gripper.pos": cmd_pos}
            self.robot.send_action(self._format_action(pose_cmd))

            precise_sleep(max(1.0 / self.fps - (time.perf_counter() - t0), 0.0))

            obs = self.robot.get_observation()
            actual_pos = float(obs[f"{self.prefix}gripper.pos"])
            load = self._read_gripper_load()

            pos_lag = actual_pos - cmd_pos
            jaw_velocity = abs(actual_pos - prev_pos)
            prev_pos = actual_pos
            jaw_stopped = jaw_velocity < 0.35

            if actual_pos <= self.empty_closed_pos and jaw_stopped:
                self.last_grasp_success = False
                self._set_gripper_torque_limit(1000)
                return actual_pos

            resisting = jaw_stopped and (
                pos_lag >= self.stall_error_deg or load >= self.load_threshold
            )
            if resisting:
                firm_ticks += 1
                if firm_ticks >= self.confirm_ticks:
                    self.last_grasp_success = True
                    latched_pos = max(self.empty_closed_pos, actual_pos - 1.5)
                    self.robot.send_action(self._format_action({**current, "gripper.pos": latched_pos}))
                    return latched_pos
            else:
                firm_ticks = 0

        self.last_grasp_success = actual_pos > self.empty_closed_pos
        return actual_pos

    # ------------------------------------------------------------------
    # Trajectory Execution Engine
    # ------------------------------------------------------------------
    def _run_waypoints(
        self,
        waypoints: list[tuple[dict, float]],
        current: dict[str, float],
        pan_override: Optional[float] = None,
        ignore_stop: bool = False,
    ) -> dict[str, float]:
        joint_keys = list(current.keys())
        peak_factor = math.pi / 2.0

        for target, min_duration in waypoints:
            if not ignore_stop and self._stop_event.is_set():
                return current

            if target.get("gripper.pos") == GRASP:
                arm_only = {k: v for k, v in target.items() if k != "gripper.pos"}
                if arm_only:
                    current = self._run_waypoints([(arm_only, min_duration)], current, pan_override, ignore_stop)

                latched_grip = self._execute_torque_grasp(current)
                current["gripper.pos"] = latched_grip

                if not self.last_grasp_success:
                    raise _EmptyGraspAbort(current)
                continue

            goal = {k: target.get(k, current[k]) for k in joint_keys}
            if pan_override is not None and "shoulder_pan.pos" in goal:
                goal["shoulder_pan.pos"] = float(pan_override)

            if "gripper.pos" in goal and goal["gripper.pos"] > (current["gripper.pos"] + 15.0):
                self._set_gripper_torque_limit(1000)

            max_delta = max((abs(goal[k] - current[k]) for k in joint_keys), default=0.0)
            safe_duration = (peak_factor * max_delta / self.max_speed) if self.max_speed > 0 else 0.0
            duration = max(min_duration, safe_duration)

            steps = max(int(math.ceil(duration * self.fps)), 1)
            for i in range(1, steps + 1):
                if not ignore_stop and self._stop_event.is_set():
                    return current
                t0 = time.perf_counter()
                alpha = 0.5 * (1.0 - math.cos(math.pi * (i / steps)))
                interp = {k: current[k] + alpha * (goal[k] - current[k]) for k in joint_keys}
                self.robot.send_action(self._format_action(interp))
                precise_sleep(max(1.0 / self.fps - (time.perf_counter() - t0), 0.0))

            current = goal

        return current

    def _step_smooth_pan_sweep(self, dt: float) -> None:
        """Updates shoulder_pan.pos with braking-limited PD velocity smoothing."""
        now = time.perf_counter()
        with self._pan_lock:
            angle_deg = self._visual_angle_deg
            pan = self.hold_pose["shoulder_pan.pos"]
            stow_pan = self.base_hold_pose["shoulder_pan.pos"]
            stale = self._last_seen_time is not None and (now - self._last_seen_time) > 0.25

            if angle_deg is not None and not stale:
                err_mag = abs(angle_deg)
                if err_mag <= self.pan_deadband_deg:
                    target_vel = 0.0
                else:
                    # PD control with kinematic stopping-distance speed cap to eliminate overshoot
                    pd_vel = self.pan_sign * (self.pan_kp * angle_deg + self.pan_kd * self._angle_rate_deg_s)
                    braking_dist = max(0.0, err_mag - 0.5 * self.pan_deadband_deg)
                    max_brake_speed = math.sqrt(2.0 * self.sweep_max_decel * braking_dist)
                    speed_cap = min(self.sweep_max_speed, max_brake_speed)
                    target_vel = max(-speed_cap, min(speed_cap, pd_vel))
            else:
                grace_expired = (
                    self._last_seen_time is not None
                    and (now - self._last_seen_time) >= self.lost_grace_sec
                )
                should_return = self._force_return_to_stow or (self.return_to_stow_on_lost and grace_expired)

                if should_return:
                    pan_err = stow_pan - pan
                    err_mag = abs(pan_err)
                    if err_mag <= 0.35:
                        target_vel = 0.0
                        pan = stow_pan
                        self._force_return_to_stow = False
                        self._last_seen_time = None
                    else:
                        raw_vel = self.pan_kp * pan_err
                        max_brake_speed = math.sqrt(2.0 * self.sweep_max_decel * err_mag)
                        speed_cap = min(self.sweep_max_speed, max_brake_speed)
                        target_vel = max(-speed_cap, min(speed_cap, raw_vel))
                else:
                    target_vel = 0.0

            # Use higher deceleration when braking or reversing direction so pan never overshoots
            is_braking_or_reversing = (
                abs(target_vel) < abs(self._current_pan_vel)
                or (target_vel * self._current_pan_vel) < 0.0
            )
            accel_limit = self.sweep_max_decel if is_braking_or_reversing else self.sweep_max_accel
            max_dv = accel_limit * dt

            vel_delta = target_vel - self._current_pan_vel
            self._current_pan_vel += max(-max_dv, min(max_dv, vel_delta))

            if abs(self._current_pan_vel) < 0.05 and target_vel == 0.0:
                self._current_pan_vel = 0.0

            pan += self._current_pan_vel * dt
            pan = max(self.pan_limits_deg[0], min(self.pan_limits_deg[1], pan))
            self.hold_pose["shoulder_pan.pos"] = pan

    def _control_loop(self) -> None:
        if self.stow_on_start:
            try:
                obs = self.robot.get_observation()
                start_pose = {k: float(obs[f"{self.prefix}{k}"]) for k in self.hold_pose.keys()}
                self._run_waypoints([(self.base_hold_pose, 1.2)], start_pose, pan_override=None)
            except Exception as e:
                print(f"[ArmHoldController] Startup stow interpolation note: {e}")

        dt = 1.0 / self.fps
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                sequence, bin_index, pan_angle, return_to_hold, done_event = self._cmd_queue.get_nowait()
                self._busy_event.set()
                try:
                    obs = self.robot.get_observation()
                    current = {k: float(obs[f"{self.prefix}{k}"]) for k in self.hold_pose.keys()}

                    current = self._run_waypoints(sequence, current, pan_override=pan_angle)

                    if not self._stop_event.is_set():
                        bin_seq = self.bin_waypoints[bin_index]
                        if isinstance(bin_seq, tuple):
                            bin_seq = [bin_seq]
                        current = self._run_waypoints(bin_seq, current, pan_override=None)

                    if return_to_hold and not self._stop_event.is_set():
                        self._set_gripper_torque_limit(1000)
                        with self._pan_lock:
                            self.hold_pose = dict(self.base_hold_pose)
                            self._visual_angle_deg = None
                            self._prev_visual_angle_deg = None
                            self._angle_rate_deg_s = 0.0
                            self._last_seen_time = None
                            self._force_return_to_stow = False
                            self._current_pan_vel = 0.0
                        self._run_waypoints([(self.hold_pose, 1.0)], current, pan_override=None)

                except _EmptyGraspAbort as abort:
                    print("[ArmHoldController] Empty grasp detected — aborting lift & bin drop.")
                    self._set_gripper_torque_limit(1000)
                    with self._pan_lock:
                        self.hold_pose = dict(self.base_hold_pose)
                        self._visual_angle_deg = None
                        self._prev_visual_angle_deg = None
                        self._angle_rate_deg_s = 0.0
                        self._last_seen_time = None
                        self._force_return_to_stow = False
                        self._current_pan_vel = 0.0
                    if not self._stop_event.is_set():
                        self._run_waypoints([(self.hold_pose, 1.0)], abort.current_pose, pan_override=None)
                finally:
                    if self._cmd_queue.empty():
                        self._busy_event.clear()
                    done_event.set()
            except queue.Empty:
                self._step_smooth_pan_sweep(dt)
                with self._pan_lock:
                    pose_cmd = dict(self.hold_pose)
                self.robot.send_action(self._format_action(pose_cmd))
                precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
import mujoco
import mujoco.viewer
import time
import numpy as np

XML_PATH = "ur5e.xml"
Q_HOME = np.array([0, -1.57, 1.57, -1.57, -1.57, 0]) 
Q_DROP = np.array([-1.2, -1.2, 1.8, -2.1, -1.57, 1.57]) # Vị trí thả mục tiêu

class RobotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) 
                             for n in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
        self.suction_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "suction_actuator")
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "suction_point")
        self.Kp, self.Kd = 1200.0, 80.0 

    def solve_ik(self, target_pos, angle_deg=0):
        q_backup = self.data.qpos.copy()
        angle_rad = np.deg2rad(angle_deg)
        for _ in range(40):
            current_pos = self.data.site_xpos[self.site_id]
            pos_error = target_pos - current_pos
            current_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
            
            z_axis = current_mat[:, 2]
            target_z = np.array([0, 0, -1])
            rot_error_z = np.cross(z_axis, target_z)
            
            target_x = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])
            x_axis = current_mat[:, 0]
            rot_error_x = np.cross(x_axis, target_x)
            
            rot_error = rot_error_z + rot_error_x
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            jac = np.vstack([jacp, jacr])
            full_error = np.concatenate([pos_error, rot_error])
            reg = 1e-4 * np.eye(6)
            dq = jac.T @ np.linalg.inv(jac @ jac.T + reg) @ full_error
            self.data.qpos[:6] += dq[:6] * 0.4 
            mujoco.mj_forward(self.model, self.data)
            if np.linalg.norm(pos_error) < 0.001: break
        
        result_q = self.data.qpos[:6].copy()
        self.data.qpos[:] = q_backup 
        mujoco.mj_forward(self.model, self.data)
        return result_q

    def control_step(self, target_qpos, suction_on):
        torque = self.Kp * (target_qpos - self.data.qpos[:6]) - self.Kd * self.data.qvel[:6]
        torque += self.data.qfrc_bias[:6]
        for i, act_id in enumerate(self.actuator_ids):
            if act_id != -1: self.data.ctrl[act_id] = torque[i]
        if self.suction_id != -1: self.data.ctrl[self.suction_id] = 1.0 if suction_on else 0.0

def main():
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    d.qpos[:6] = Q_HOME
    mujoco.mj_forward(m, d)
    
    controller = RobotController(m, d)
    current_target = Q_HOME.copy()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t_cycle = d.time % 12.0 
            
            cube_pos = d.body("cube_red").xpos.copy()
            desired_q, suction = Q_HOME, False

            # --- LOGIC CHU KỲ (Fix lỗi cờ biến) ---
            if t_cycle < 2.0: 
                desired_q = Q_HOME # Đợi ở vị trí nghỉ
            elif t_cycle < 4.5: 
                # Hạ xuống hút vật (Góc 0 độ)
                desired_q = controller.solve_ik(cube_pos + np.array([0, 0, 0.01]), angle_deg=0)
                if t_cycle > 4.0: suction = True
            elif t_cycle < 6.5:
                # Nhấc lên và xoay 90 độ
                desired_q = controller.solve_ik(cube_pos + np.array([0, 0, 0.01]), angle_deg=90)
                suction = True
            elif t_cycle < 9.0:
                # Di chuyển đến vị trí thả (Vẫn giữ suction và góc 90)
                desired_q = Q_DROP 
                suction = True
            elif t_cycle < 10.5:
                # Thả vật xuống
                desired_q = Q_DROP
                suction = False 
            else:
                # Quay về Home để chuẩn bị Reset
                desired_q = Q_HOME
                suction = False

            # --- RESET VẬT VÀ TAY MÁY (Quan trọng nhất) ---
            if t_cycle > 11.8:
                # Reset khối vuông về vị trí gốc trên bàn (x=0.5, y=0.0, z=0.03)
                j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
                if j_id != -1:
                    d.qpos[m.jnt_qposadr[j_id] : m.jnt_qposadr[j_id]+3] = [0.52, 0.1375, 0.025]
                    d.qvel[m.jnt_dofadr[j_id] : m.jnt_dofadr[j_id]+6] = 0
                
                # Reset tay máy về Home tức thì để bắt đầu chu kỳ mới
                d.qpos[:6] = Q_HOME
                d.qvel[:6] = 0
                current_target = Q_HOME.copy()
                mujoco.mj_forward(m, d)

            # Nội suy mượt
            diff = desired_q - current_target
            current_target += np.clip(diff, -0.04, 0.04)
            
            controller.control_step(current_target, suction)
            mujoco.mj_step(m, d)
            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
# coding: utf-8


from scipy.stats import expon, norm, uniform

# roboto.pyの内容を全てimport
from robot import *


# ロボットの雑音を実装する
# ロボット自体の機能はIdealRobotを継承
class Robot(IdealRobot):


    # noise_per_meter : 1メートルあたりの小石の数
    # noise_std : 小石を踏んだ場合に発生するロボットの方向に対する雑音の標準偏差
    # bias_rate_stds : ロボットに加えるバイアス
    # expected_stuck_time : 引っかかりにかかるまでの時間の期待値
    # expected_escape_time : 引っかかった後、脱出するまでの時間の期待値
    def __init__(self, pose, agent = None, sensor = None, color = "black", \
            noise_per_meter = 5, noise_std = math.pi / 60, \
            bias_rate_stds = (0.1, 0.1), \
            expected_stuck_time = 1e100, expected_escape_time = 1e-100,\
            expected_kidnap_time = 1e100, kidnap_range_x = (-5.0, 5.0), kidnap_range_y = (-5.0, 5.0)):

        super().__init__(pose, agent, sensor, color)

        # ノイズに関するパラメータ
        # ゼロで割るのを防止
        self.noise_pdf = expon(scale = 1.0 / (1e-100 + noise_per_meter))
        
        # どれくらいの距離を進んだらノイズが加わるか
        # 指数分布の確率変数
        self.distance_until_noise = self.noise_pdf.rvs()

        # scale : 小石を踏むまでの平均の道のり
        self.theta_noise = norm(scale = noise_std)

        # バイアス関するパラメータ
        # 速度に加えるバイアス
        self.bias_rate_nu = norm.rvs(loc = 1.0, scale = bias_rate_stds[0])
        # 角速度に加えるバイアス
        self.bias_rate_omega = norm.rvs(loc = 1.0, scale = bias_rate_stds[1])

        # 引っかかりに関するパラメータ
        self.stuck_pdf = expon(scale = expected_stuck_time)
        self.escape_pdf = expon(scale = expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False

        # ロボットの誘拐に関するパラメータ
        self.kidnap_pdf = expon(scale = expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc = (rx[0], ry[0], 0.0), scale = (rx[1] - rx[0], ry[1] - ry[0], 2 * math.pi))

    
    # ロボットの誘拐の実装
    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose


    # 引っかかりの実装
    # 引っかかるとロボットが動かなくなる実装
    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu * (not self.is_stuck), omega * (not self.is_stuck)


    # バイアスを加える
    # 毎回一定のズレを与える
    def bias(self, nu, omega):
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega


    # omegaにノイズを加える
    def noise(self, pose, nu, omega, time_interval):
        # 一定の距離を走ったらロボットが曲がる
        self.distance_until_noise -= nu * time_interval + self.r * omega * time_interval
        # 進んだ距離が一定の距離に達した場合
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()

        return pose


    # 1フレームごとの計算
    def one_step(self, time_interval):
        if not self.agent:
            return 
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        # ロボットにバイアスを加える
        nu, omega = self.bias(nu, omega)
        # ロボットの引っかかりを再現
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)

        if self.sensor:
            self.sensor.data(self.pose)


# カメラの観測結果にノイズを加える
class Camera(IdealCamera):

    
    # distance_noise_rate : 距離に加える雑音の標準偏差の割合
    # direction_noise : 角度に加える雑音の標準偏差の割合
    # distnce_bias_rate_stddev : 距離のバイアスの量を決定する正規分布の標準偏差
    # direction_bias_stddev : 角度のバイアスの量を決定する正規分布の標準偏差
    def __init__(self, env_map, distance_range = (-0.5, 6.0), \
            direction_range = (-math.pi / 3, math.pi / 3), \
            distance_noise_rate = 0.1, direction_noise = math.pi / 90,\
            distance_bias_rate_stddev = 0.1, direction_bias_stddev = math.pi / 90,\
            phantom_prob = 0.0, phantom_range_x = (-5.0, 5.0), phantom_range_y = (-5.0, 5.0),\
            oversight_prob = 0.1):

        super().__init__(env_map, distance_range, direction_range)
        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

        self.distance_bias_rate_std = norm.rvs(scale = distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale = direction_bias_stddev)

        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc = (rx[0], ry[0]), scale = (rx[1] - rx[0], ry[1] - ry[0]))
        self.phantom_prob = phantom_prob

        self.oversight_prob = oversight_prob


    # 見落としを実装
    def oversight(self, relpos):
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos


    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            # 偽のランドマークを計算
            pos = np.array(self.phantom_dist.rvs()).T
            return self.relative_polar_pos(cam_pose, pos)
        else:
            return relpos


    # カメラの計測結果に加えるノイズの算出
    def noise(self, relpos):
        ell = norm.rvs(loc = relpos[0], scale = relpos[0] * self.distance_noise_rate)
        phi = norm.rvs(loc = relpos[1], scale = self.direction_noise)
        return np.array([ell, phi]).T


    def bias(self, relpos):
        return relpos + np.array([relpos[0] * self.distance_bias_rate_std, self.direction_bias]).T

    
    # カメラの検出結果にノイズを加える
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.relative_polar_pos(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))
        self.lastdata = observed
        return observed


if __name__ == "__main__":

    world = World(30, 0.1)

    m = Map()
    m.append_landmark(Landmark(-4, 2))
    m.append_landmark(Landmark(2, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0 / 180 * math.pi)
    r = Robot(np.array([2, 2, math.pi / 6]).T, sensor = Camera(m), agent = circling)
    world.append(r)
    world.draw()

    
    '''
    world = World(30, 0.1)

    circling = Agent(0.2, 10.0 / 180 * math.pi)

    for i in range(100):
        r = Robot(np.array([0, 0, 0]).T, agent = circling, sensor = None, color = "gray", noise_per_meter = 0, bias_rate_stds = (0.0, 0.0), expected_kidnap_time = 5)
        world.append(r)

    r = IdealRobot(np.array([0, 0, 0]).T, sensor = None, agent = circling, color = "red")
    world.append(r)

    world.draw()
    '''

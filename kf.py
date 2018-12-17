# coding: utf-8
# カルマンフィルタ p107から


from robot import *
from noise_robot import *
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


class KalmanFilter:


    # envmap : 地図
    # init_pose : 初期姿勢
    # motion_noise_stds : 動きに加えるノイズの標準偏差
    def __init__(self, envmap, init_pose, motion_noise_stds, distance_dev_rate = 0.14, direction_dev = 0.05):

        # 多次元のガウス分布の生成
        # mean : 平均値 : [0.0, 0.0, math.pi / 4]
        # cov : 共分散行列 [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.01]]
        self.belief = multivariate_normal(mean = init_pose, cov = np.diag([1e-10, 1e-10, 1e-10]))
        
        # 地図の受け渡し
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev
        self.motion_noise_stds = motion_noise_stds

    
    def observation_update(self, observation):

        for d in observation:
            z = d[0]
            obs_id = d[1]

            # Hの計算
            # 点ランドマークの姿勢(x, y)を取得
            mx, my = self.map.landmarks[obs_id].pos
            # ガウス分布の平均値(x, y, theta)を取得
            mux, muy, mut = self.belief.mean
            # 2乗和の計算
            q = (mux - mx) ** 2 + (muy - my) ** 2
            sqrtq = np.sqrt(q)
            H = np.array([[(mux - mx) / sqrtq, (muy - my) / sqrtq, 0.0], [(my - muy) / q, (mux - mx) / q, -1.0]])
            
            # Qの計算
            # ロボットから見た点ランドマークの距離と方向を取得
            hmu = IdealCamera.relative_polar_pos(self.belief.mean, self.map.landmarks[obs_id].pos)
            distance_dev = self.distance_dev_rate * hmu[0]
            Q = np.diag(np.array([distance_dev ** 2, self.direction_dev ** 2]))

            # カルマンゲインの計算
            K = self.belief.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(self.belief.cov).dot(H.T)))
        
            # パラメータの更新
            self.belief.mean += K.dot(z - hmu)
            self.belief.cov = (np.eye(3) - K.dot(H)).dot(self.belief.cov)
            

    def motion_update(self, nu, omega, time):
        if abs(nu) < 1e-10 and abs(omega) < 1e-10:
            return
        
        v = self.motion_noise_stds
        M = np.diag([v["nn"] ** 2 * abs(nu) / time + v["no"] ** 2 * abs(omega) / time, \
                v["on"] ** 2 * abs(nu) / time + v["oo"] ** 2 * abs(omega) / time])
        t = self.belief.mean[2]
        A = time * np.array([[math.cos(t), 0.0], [math.sin(t), 0.0], [0.0, 1.0]])
        F = np.diag([1.0, 1.0, 1.0])

        if abs(omega) < 10e-5:
            F[0 : 2] = -nu * time * math.sin(t)
            F[1 : 2] = nu * time * math.cos(t)
        else:
            F[0 : 2] = nu / omega * (math.cos(t + omega * time) - math.cos(t))
            F[1 : 2] = nu / omega * (math.sin(t + omega * time) - math.sin(t))

        self.belief.cov = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T)
        self.belief.mean = IdealRobot.state_transition(nu, omega, time, self.belief.mean) 


    # 楕円を描画する
    def draw(self, ax, elems):
        # 共分散行列の固有値と固有ベクトルを計算
        eig_vals, eig_vec = np.linalg.eig(self.belief.cov[0 : 2, 0 : 2])

        # ベクトルの間の角度を計算(ラジアン)
        ang = math.atan2(eig_vec[:, 0][1], eig_vec[:, 0][0]) / math.pi * 180

        # matplotで楕円を描くオブジェクトを返す
        e = Ellipse(self.belief.mean[0 : 2], width = 3 * eig_vals[0], \
                height = 3 * eig_vals[1], angle = ang, fill = False, color = "blue", alpha = 0.5)
        elems.append(ax.add_patch(e))

        #  楕円に線を描画
        x, y, c = self.belief.mean
        sigma3 = math.sqrt(self.belief.cov[2, 2]) * 3
        xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
        ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
        elems += ax.plot(xs, ys, color = "blue", alpha = 0.5)


# エージェントにカルマンフィルタを実装する
class KfAgent(Agent):


    def __init__(self, time_interval, nu, omega, \
            init_pose, envmap, motion_noise_stds = {"nn" : 0.19, "no" : 0.001, "on" : 0.13, "oo" : 0.2}):
        super().__init__(nu, omega)
        self.kf = KalmanFilter(envmap, init_pose, motion_noise_stds)
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0


    # 図の描画
    def draw(self, ax, elems):
        self.kf.draw(ax, elems)


    # return : 速度と角速度
    def decision(self, observation = None):
        self.kf.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.kf.observation_update(observation)
        return self.nu, self.omega


if __name__ == "__main__":
    
    # 制御の周期
    time_interval = 0.1

    # 世界座標系の作成
    # 30秒間シミュレーション
    world = World(30, time_interval)


    # 地図の作成
    m = Map()

    # 地図に点ランドマークを追加
    m.append_landmark(Landmark(-4, 2))
    m.append_landmark(Landmark(2, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    # robot1を作成
    circling = KfAgent(time_interval, 0.2, 10.0 / 180.0 * math.pi, np.array([0, 0, 0]).T, m)
    r = Robot(np.array([0, 0, 0]).T, sensor = Camera(m), agent = circling, color = "red")
    world.append(r)

    # robot2を作成
    linear = KfAgent(time_interval, 0.1, 0.0, np.array([0, 0, 0]).T, m)
    r = Robot(np.array([0, 0, 0]).T, sensor = Camera(m), agent = linear, color = "red")
    world.append(r)

    # robot3を作成
    right = KfAgent(time_interval, 0.1, -3.0 / 180 * math.pi, np.array([0, 0, 0]).T, m)
    r = Robot(np.array([0, 0, 0]).T, sensor = Camera(m), agent = right, color = "red")
    world.append(r)

    # 図を描画
    world.draw()

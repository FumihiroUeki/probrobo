# coding: utf-8


import sys
from robot import *
from noise_robot import *
from scipy.stats import multivariate_normal 
import random


class Particle:


    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight


    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        pnu = nu + ns[0] * math.sqrt(abs(nu) / time) + ns[1] * math.sqrt(abs(omega) / time)
        pomega = omega + ns[2] * math.sqrt(abs(nu) / time) + ns[3] * math.sqrt(abs(omega) / time)
        self.pose = IdealRobot.state_transition(pnu, pomega, time, self.pose)


    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            # パーティクルフィルタの位置と地図からランドマークの距離と方角を計算
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.relative_polar_pos(self.pose, pos_on_map)

            # 尤度の計算
            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev **2, direction_dev ** 2]))
            self.weight *= multivariate_normal(mean = particle_suggest_pos, cov = cov).pdf(obs_pos)


class Mcl:


    def __init__(self, envmap, init_pose, num, motion_noise_stds, distance_dev_rate = 0.14, direction_dev = 0.05):
        self.particles = [Particle(init_pose, 1.0 / num) for i in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev
        v = motion_noise_stds
        c = np.diag([v["nn"] ** 2, v["no"] ** 2, v["on"] ** 2, v["oo"] ** 2])
        self.motion_noise_rate_pdf = multivariate_normal(cov = c)
        self.ml_pose = self.particles[0].pose

    
    def set_ml_pose(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml_pose = self.particles[i].pose


    def motion_update(self, nu, omega, time):
        #print(self.motion_noise_rate_pdf.cov)
        for p in self.particles:
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)


    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml_pose()
        self.resampling()


    def resampling(self):
        ws = [e.weight for e in self.particles]

        if sum(ws) < 1e-100:
            ws = [e + 1e-100 for e in ws]

        ps = random.choices(self.particles, weights = ws, k = len(self.particles))

        self.particles = [Particle(e.pose, 1.0 / len(self.particles)) for e in ps]


    def draw(self, ax, elems):

        # 全パーティクルの座標を保管
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]

        # 全パーティクルの方向を保管
        vxs = [math.cos(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2]) * p.weight * len(self.particles)  for p in self.particles]

        elems.append(ax.quiver(xs, ys, vxs, vys, angles = "xy", scale_units ="xy", scale = 1.5, color = "blue", alpha = 0.5))


class MclAgent(Agent):


    def __init__(self, time_interval, nu, omega, particle_pose, envmap, particle_num = 100,\
            motion_noise_stds = {"nn" : 0.19, "no" : 0.001, "on" : 0.13, "oo" : 0.2}):
        super().__init__(nu, omega)
        self.pf = Mcl(envmap, particle_pose, particle_num, motion_noise_stds)

        # エージェントがMCLを行う周期
        self.time_interval = time_interval
        self.prev_nu = 0.0
        self.prev_omega = 0.0


    def draw(self, ax, elems):
        self.pf.draw(ax, elems)
        x, y, t = self.pf.ml_pose
        s = "({:.2f}, {:.2f}, {})".format(x, y, int(t * 180 / math.pi) % 360)
        elems.append(ax.text(x, y + 0.1, s, fontsize = 8))


    def decision(self, observation = None):
        self.pf.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.pf.observation_update(observation)
        return self.nu, self.omega


if __name__ == "__main__":
    
    time_interval = 0.1
    world = World(40, time_interval)

    m = Map()
    m.append_landmark(Landmark(-4, 2))
    m.append_landmark(Landmark(2, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    #initial_pose = np.array([2, 2, math.pi / 6]).T
    circling = MclAgent(time_interval, 0.2, 10.0 / 180*math.pi, np.array([0, 0, 0]).T, m, particle_num = 100)
    r = Robot(np.array([0, 0, 0]).T, sensor = Camera(m), agent = circling, color = "red")
    world.append(r)

    world.draw()


    '''
    initial_pose = np.array([0, 0, 0]).T

    a = MclAgent(0.1, 0.2, 10 / 180 * math.pi, initial_pose)
    a.mcl.motion_update(0.2, 10 / 180 * math.pi, 0.1)
    for p in a.mcl.particles:
        print(p.pose)
    '''

# coding: utf-8


import matplotlib
#matplotlib.use("nbagg")
from matplotlib import pyplot as plt
from matplotlib import animation as anm
from matplotlib import patches as patches
import math
import numpy as np


# 世界座標系を表示するクラス
class World:

    
    # time_span : 
    # time_interval : 
    # debug : Falseにすると図を表示しない
    def __init__(self, time_span, time_interval, debug = False):

        # ロボットのオブジェクトを追加するリスト
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval

    
    # ロボットのオブジェクトを追加する関数 
    # ojb : ロボットのオブジェクト
    def append(self, obj):
        self.objects.append(obj)


    # matplotlibでロボット等を一括して表示する
    def draw(self):

        # 8x8で図を表示する
        fig = plt.figure(figsize = (6, 6)) 
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        # x軸の表示範囲
        ax.set_xlim(-5, 5)
        # y軸の表示範囲
        ax.set_ylim(-5, 5)
        ax.set_title("${Robot-Simulator}$", fontsize = 10)
        ax.set_xlabel("X", fontsize = 20)
        ax.set_ylabel("Y", fontsize = 20)

        elems = []

        if self.debug:
            for i in range(1000):
                self.one_sptep(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, \
                    fargs = (elems, ax), frames = int(self.time_span / self.time_interval) + 1, \
                    interval = int(self.time_interval * 1000), repeat = False)
            plt.show()

    
    # アニメーションを表示する際に
    # 毎フレームおきに呼ばれる関数
    # i : フレーム数
    # elems : 表示する図形のリスト
    # ax : サブプロット
    def one_step(self, i, elems, ax):
        while elems:
            # 前フレームの内容をクリアする
            elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval * i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize = 10))
        for obj in self.objects:
            # elemsにある各図形を表示
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)


# ロボットを定義するクラ
class IdealRobot:

    
    # pose: ロボットの姿勢 [x, y, theata] (注意 : thetaはラジアンで与える)
    # agent : ロボットを操作するオブジェクト
    # sensor : ロボットのセンサ 下にあるIdealCameraのこと
    # color : ロボットの色
    def __init__(self, pose, agent = None, sensor = None, color = "black"):

        self.pose = pose
        # ロボットを表示する際の半径
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.sensor = sensor
        self.poses = [pose]


    # ロボットに回転や移動の動作を実装
    # nu : ロボットの速度
    # time : 動作時間
    # pose : ロボットの前の時間の姿勢
    @classmethod
    def state_transition(cls, nu, omega, time, pose):

        t0 = pose[2]
       
        # omegaが0の場合の計算式
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu * math.cos(t0), nu * math.sin(t0), omega]) * time
        else:
            return pose + np.array([nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)), \
                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)), \
                    omega * time])


    # ロボットの表示をする際に
    # 毎フレーム呼ばれる関数
    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        # エージェントから速度と角加速度を受け取る
        nu, omega = self.agent.decision(obs)
        # 姿勢を更新
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


    # ロボットを実際に表示させる関数
    def draw(self, ax, elems):
        # x, y : ロボットの中心座標
        # theta : ロボットの向いている方向
        x, y, theta = self.pose
        # ロボットがどの方向を向いているかを
        # 視覚的にわかりやすくするために
        # 線を表示する
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color = self.color)
        # ロボット自体の円を表示する
        c = patches.Circle(xy = (x, y), radius = self.r, fill = False, color = self.color)
        elems.append(ax.add_patch(c))

        # ロボットが走った軌跡を表示する
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], \
                [e[1] for e in self.poses], linewidth = 0.5, color = "black")
        
        # ロボットにカメラが搭載されている場合
        if self.sensor:
            # IdealCameraのdata関数を呼び出す
            self.sensor.draw(ax, elems, self.poses[-2])

        # ロボットにエージェントがあり、draw関数がある場合
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)



# ロボットを操作するクラス
class Agent:

    
    # nu : ロボットの速度
    # omega : ロボットの角速度
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega


    def decision(self, observation = None):
        return self.nu, self.omega


# 点ランドマークを表示するクラス
class Landmark:

    
    # x, y : ランドマークの座標
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None


    # ランドマークを表示する
    def draw(self, ax, elems):
        # ランドマークを*で表示する
        c = ax.scatter(self.pos[0], self.pos[1], s = 100, \
                marker = "*", label = "landmarks", color = "orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id : " + str(self.id), fontsize = 10))


# 地図
class Map:


    def __init__(self):
        # ランドマークを追加するリスト
        self.landmarks = []


    # ランドマークを追加する
    def append_landmark(self, landmark):
        # 追加するランドマークにIDを与える
        landmark.id = len(self.landmarks) 
        self.landmarks.append(landmark)


    # ランドマークを表示する
    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)


# ロボットのカメラを実装
class IdealCamera:


    # env_map : 地図のオブジェクト
    def __init__(self, env_map, distance_range = (0.5, 6.0),\
            direction_range = (-math.pi / 3, math.pi / 3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    
    # カメラの視野を表現するために
    # ランドマークを観測できる条件を設ける
    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1]\
                and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]


    # 地図内のランドーマーク全てについて計算
    # cam_pose : カメラ(ロボット)の姿勢
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.relative_polar_pos(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))

        self.lastdata = observed

        return observed


    # ロボットから見たランドマークの距離と方向を計算
    # cam_pose : カメラ(ロボット)の姿勢
    # obj_pos : ランドマークの位置
    @classmethod
    def relative_polar_pos(cls, cam_pose, obj_pos):
        s = math.sin(cam_pose[2])
        c = math.cos(cam_pose[2])

        # 世界座標系からロボット座標系に変換 
        relative_pos = np.array([[c, s], [-s, c]]).dot(obj_pos - cam_pose[0 : 2])

        # ロボットからランドマークまでの距離
        distance = math.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
        # ロボットから見たランドマークの向き
        direction = math.atan2(relative_pos[1], relative_pos[0])

        return np.array([distance, direction]).T

    
    # カメラの計測値を表示する
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color = "pink")


if __name__ == "__main__":
    
    world = World(30, 0.1, debug = False)

    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0 / 180 * math.pi)

    robot1 = IdealRobot(np.array([2, 3, math.pi / 6]).T, sensor = IdealCamera(m), agent = straight)
    robot2 = IdealRobot(np.array([-2, 1, math.pi / 5 * 6]).T, sensor = IdealCamera(m), agent = circling, color = "red")

    world.append(robot1)
    world.append(robot2)

    world.draw()

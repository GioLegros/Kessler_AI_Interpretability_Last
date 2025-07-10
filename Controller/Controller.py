import math
from kesslergame import KesslerController, Ship
import numpy as np
from typing import Dict, Tuple, Any
from func import ast_angle, angle360

N_pop = 30
N_rep = 10
count_max = 10000
p = 0.9
K = 3


def switch_rule(x):
    if x == [0, 0]: return [4,1]
    elif x == [0, 1]: return [3,1]
    elif x == [0, 2]: return [2,2]
    elif x == [0, 3]: return [3,3]
    elif x == [0, 4]: return [4,3]
    elif x == [1, 0]: return [4,0]
    elif x == [1, 1]: return [5,1]
    elif x == [1, 2]: return [5,1]
    elif x == [1, 3]: return [5,3]
    elif x == [1, 4]: return [4,3]
    elif x == [2, 0]: return [4,0]
    elif x == [2, 1]: return [5,1]
    elif x == [2, 2]: return [5,1]
    elif x == [2, 3]: return [5,3]
    elif x == [2, 4]: return [4,3]
    elif x == [3, 0]: return [4,0]
    elif x == [3, 1]: return [5,1]
    elif x == [3, 2]: return [5,1]
    elif x == [3, 3]: return [5,3]
    elif x == [3, 4]: return [4,3]
    elif x == [4, 0]: return [5,1]
    elif x == [4, 1]: return [5,1]
    elif x == [4, 2]: return [5,2]
    elif x == [4, 3]: return [5,3]
    elif x == [4, 4]: return [4,3]
    else: return None

def outs(out):
    out_speed = np.array([-1.00, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.00])
    out_angle = np.array([-180, -180, 0, 180, 180])
    out_speed = out_speed[out[0]]*450
    out_angle = out_angle[out[1]]
    return np.array([out_speed, out_angle])

def membership_dist(x):
    x /= 400
    b = [0.0, 0.25, 0.5, 0.75, 1.0]
    if x <= b[0]:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif x <= b[1]:
        return ([1.0 - (x - b[0]) / (b[1] - b[0]), (x - b[0]) / (b[1] - b[0]), 0.0, 0.0, 0.0])
    elif x <= b[2]:
        return ([0.0, (b[2] - x) / (b[2] - b[1]),
                         (x - b[1]) / (b[2] - b[1]), 0.0, 0.0])
    elif x <= b[3]:
        return ([0.0, 0.0, (b[3] - x) / (b[3] - b[2]), (x - b[2]) / (b[3] - b[2]), 0.0])
    elif x <= b[4]:
        return ([0.0, 0.0, 0.0, (b[4] - x) / (b[4] - b[3]), (x - b[3]) / (b[4] - b[3])])
    else:
        return ([0.0, 0.0, 0.0, 0.0, 1.0])


def membership_out1(x):
    x /= 450
    b = [-1.00, -0.75, -0.5, -0.25, 0, 0.25, 0.50, 0.75, 1.00]
    if x <= b[0]:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif x <= b[1]:
        return np.array(
            [1.0 - (x - b[0]) / (b[1] - b[2]), (x - b[0]) / (b[1] - b[0]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif x <= b[2]:
        return np.array([0.0, (b[2] - x) / (b[2] - b[1]),
                         (x - b[1]) / (b[2] - b[1]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elif x <= b[3]:
        return np.array([0.0, 0.0, (b[3] - x) / (b[3] - b[2]), (x - b[2]) / (b[3] - b[2]), 0.0, 0.0, 0.0, 0.0, 0.0])
    elif x <= b[4]:
        return np.array([0.0, 0.0, 0.0, (b[4] - x) / (b[4] - b[3]), (x - b[3]) / (b[4] - b[3]), 0.0, 0.0, 0.0, 0.0])
    elif x <= b[5]:
        return np.array([0.0, 0.0, 0.0, 0.0, (b[5] - x) / (b[5] - b[4]), (x - b[4]) / (b[5] - b[4]), 0.0, 0.0, 0.0])
    elif x <= b[6]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, (b[6] - x) / (b[6] - b[5]), (x - b[5]) / (b[6] - b[5]), 0.0, 0.0])
    elif x <= b[7]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (b[7] - x) / (b[7] - b[6]), (x - b[6]) / (b[7] - b[6]), 0.0])
    elif x <= b[8]:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (b[8] - x) / (b[8] - b[7]), (x - b[7]) / (b[8] - b[7])])
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def membership2(x, b=None):
    x /= 180
    if b is None:
        b = [-1.0, -0.5, 0, 0.50, 1.0]

    if x <= b[0]:
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    elif x <= b[1]:
        return [1.0 - (x - b[0]) / (b[1] - b[2]), (x - b[0]) / (b[1] - b[0]), 0.0, 0.0, 0.0]
    elif x <= b[2]:
        return ([0.0, (b[2] - x) / (b[2] - b[1]),
                 (x - b[1]) / (b[2] - b[1]), 0.0, 0.0])
    elif x <= b[3]:
        return [0.0, 0.0, (b[3] - x) / (b[3] - b[2]), (x - b[2]) / (b[3] - b[2]), 0.0]
    elif x <= b[4]:
        return [0.0, 0.0, 0.0, (b[4] - x) / (b[4] - b[3]), (x - b[3]) / (b[4] - b[3])]
    else:
        return [0.0, 0.0, 0.0, 0.0, 1.0]


def out_speed(x, speed_bound=None):
    x = np.array(x[:5])
    speed_bound = np.array([-480, -240, 0, 240, 480])
    return np.inner(x[:, 0], speed_bound)


def mems(X):
    out = [[membership_dist(x[0]), membership2(x[1])] for x in X]
    return out
def mem(x):
    out = [membership_dist(x[0]), membership2(x[1])]
    return out


def mems2(y):
    out = [[membership_out1(a[0]), membership2(a[1])] for a in y]
    return out
def mu(x):
    out = np.outer(x[0], x[1])
    return out

def conf(x):
    C_dist = np.argmax(x[0])
    C_angle = np.argmax(x[1])
    C = np.array([C_dist, C_angle])
    return C.tolist()

def output(M, rule):
    out = np.sum(M * rule)
    return out

def conf_out(y):
    mem_out_dist = np.array([membership_out1(a[0]) for a in y])
    mem_out_angle = np.array([membership2(a[1]) for a in y])
    C_out = np.argmax(mem_out_dist, axis=1)
    C_out_Angle = np.argmax(mem_out_angle, axis=1)
    C_out = np.array([C_out, C_out_Angle]).T
    return C_out.tolist()


class Controller(KesslerController):
    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
    """


    def __init__(self, *args):
        pass

    def actions(self, ownship: Dict, input_data: Dict[str, Tuple]) -> tuple[int | float, int | float, bool | Any, bool]:

        ast_list = np.array(input_data["asteroids"])
        dist_xylist = [np.array(ownship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoid_list = dist_xylist.copy()
        dist_list1 = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_xylist]


        for xy in dist_avoid_list:
            if xy[0] > 500:
                xy[0] -= 1000
            elif xy[0] < -500:
                xy[0] += 1000
            if xy[1] > 400:
                xy[1] -= 800
            elif xy[1] < -400:
                xy[1] += 800
        dist_avoid_list = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_avoid_list]

        sorted2_idx = np.argsort(dist_avoid_list)
        sorteddict = ast_list[sorted2_idx]

        search_list = sorteddict[0:5]
        angle_dist = [np.array(ast['position']) - np.array(ownship['position']) for ast in search_list]
        angle_dist = [angle360(math.degrees((np.arctan2(near_ang[1], near_ang[0])))) - ownship['heading'] for near_ang
                      in angle_dist]
        aalist = []
        for ang in angle_dist:
            if ang > 180:
                ang -= 360
            elif ang < -180:
                ang += 360
            aalist.append(ang)
        angdiff_front = min(aalist, key=abs)
        angdiff = aalist[0]
        fire_bullet = abs(angdiff_front) < 15 and min(dist_list1) < 400
        avoidance = np.min(dist_avoid_list)

        if len(input_data['ships']) >= 2:
            angle_ships = ast_angle(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            dist = math.dist(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            if (dist <= avoidance):
                if angle_ships > 180:
                    angle_ships -= 360
                elif angle_ships < -180:
                    angle_ships += 360
                avoidance = dist
                angdiff = abs(angle_ships)

        print(min(dist_avoid_list))
        if min(dist_avoid_list) < 100:
            drop_mine = True
            print("ああ")
        else:
            drop_mine = False
        M = mem([avoidance, angdiff])
        MU = mu(M)
        thrust, turn_rate = 0.0, 0.0
        for i in range(5):
            for j in range(5):

                out = MU[i][j] * outs(switch_rule([i, j]))
                thrust += out[0]
                turn_rate += out[1]
        if turn_rate < -180:
            turn_rate = -180
        elif turn_rate > 180:
            turn_rate = 180

        if thrust > 480:
            thrust = 480
        elif thrust < -480:
            thrust = -480

        return thrust, turn_rate, fire_bullet, drop_mine

    @property
    def name(self) -> str:
        return "OMU-Let's"


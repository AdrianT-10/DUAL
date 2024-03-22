#!/usr/bin/env python
import heapq
import threading
import xml.etree.ElementTree as ET
import glob
import os
import sys
import csv
from threading import Lock
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
try:
    sys.path.append(glob.glob(os.path.abspath('%s/../../carla/dist/carla-*%d.%d-%s.egg' % (
        os.path.realpath(__file__),
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

os.environ["PYRO_LOGFILE"] = "pyro.log"
os.environ["PYRO_LOGLEVEL"] = "DEBUG"
import networkx as nx
from collections import defaultdict
from multiprocessing import Process
from threading import RLock
import Pyro4
import argparse
import carla
import math
import random
import time
if sys.version_info.major == 2:
    from pathlib2 import Path
else:
    from pathlib import Path
from networkx import Graph as NetworkxGraph


# 常量定义
''' ========== CONSTANTS ========== '''
DATA_PATH = Path(os.path.realpath(__file__)).parent.parent.parent/'Data'# 数据路径
PATH_MIN_POINTS = 20 # 路径最小点数
PATH_INTERVAL = 1.0 # 路径间隔

SPAWN_DESTROY_MAX_RATE = 15.0 # 最大生成销毁速率
GAMMA_MAX_RATE = 40.0  # GAMMA最大速率
CONTROL_MAX_RATE = 20.0 # 控制最大速率
COLLISION_STATISTICS_MAX_RATE = 5.0 # 碰撞统计最大速率
SPAWN_DESTROY_REPETITIONS = 3 # 生成销毁重复次数
# 全局累积权值和锁
accumulated_weight = 0
accumulated_weight_lock = Lock()

# Ziegler-Nichols调谐参数：K_p, T_u
CAR_SPEED_PID_PROFILES = {
    'vehicle.volkswagen.t2': [1.5, 25.0 / 25],
    'vehicle.carlamotors.carlacola': [3.0, 25.0 / 25],
    'vehicle.jeep.wrangler_rubicon': [1.5, 25.0 / 27],
    'vehicle.nissan.patrol': [1.5, 25.0 / 24],
    # 'vehicle.tesla.cybertruck': [3.0, 25.0 / 28],
    'vehicle.chevrolet.impala': [0.8, 20.0 / 17],
    'vehicle.audi.tt': [0.8, 20.0 / 15],
    'vehicle.mustang.mustang': [1.0, 20.0 / 15],
    'vehicle.citroen.c3': [0.7, 20.0 / 14],
    'vehicle.toyota.prius': [1.2, 20.0 / 15],
    'vehicle.dodge_charger.police': [1.0, 20.0 /17],
    'vehicle.mini.cooperst': [0.8, 20.0 / 14],
    'vehicle.audi.a2': [0.8, 20.0 / 18],
    'vehicle.nissan.micra': [0.8, 20.0 / 19],
    'vehicle.seat.leon': [0.8, 20.0 / 16],
    'vehicle.tesla.model3': [1.5, 20.0 / 16],
    'vehicle.mercedes-benz.coupe': [0.8, 20.0 / 15],
    'vehicle.lincoln.mkz2017': [1.3, 20.0 / 15],
    'vehicle.bmw.grandtourer': [1.5, 20.0 / 16],
    'default': [1.6, 25.0 / 34]
}

BIKE_SPEED_PID_PROFILES = {
    'vehicle.diamondback.century': [1.5, 20.0 / 23.0],
    'vehicle.gazelle.omafiets': [1.5, 20.0 / 23.0],
    'vehicle.bh.crossbike': [1.5, 20.0 / 23.0],
    'default': [0.75, 25.0 / 35.0]
}


CAR_STEER_PID_PROFILES = {
    'vehicle.volkswagen.t2': [2.5, 10.0 / 13],
    'vehicle.carlamotors.carlacola': [2.5, 10.0 / 15],
    'vehicle.jeep.wrangler_rubicon': [2.8, 10.0 / 17],
    'vehicle.nissan.patrol': [3.2, 10.0 / 14],
    # 'vehicle.tesla.cybertruck': [4.0, 10.0 / 16.0],
    'vehicle.audi.etron': [3.0, 10.0 / 15],
    'vehicle.chevrolet.impala': [2.5, 10.0 / 19],
    'vehicle.audi.tt': [2.3, 10.0 / 20],
    'vehicle.mustang.mustang': [2.8, 10.0/ 19],
    'vehicle.citroen.c3': [2.0, 10.0 / 17],
    'vehicle.toyota.prius': [2.1, 10.0 / 18],
    'vehicle.dodge_charger.police': [2.3, 10.0 / 21],
    'vehicle.mini.cooperst': [2.0, 10.0 / 16],
    'vehicle.audi.a2': [2.0, 10.0 / 18],
    'vehicle.nissan.micra': [3.3, 10.0 / 23],
    'vehicle.seat.leon': [2.2, 10.0 / 20],
    'vehicle.tesla.model3': [2.7, 10.0 / 19],
    'vehicle.mercedes-benz.coupe': [2.7, 10.0 / 20],
    'vehicle.lincoln.mkz2017': [2.7, 10.0 / 16],
    'vehicle.bmw.grandtourer': [2.7, 10.0/ 17],
    'default': [2.8, 10.0 / 15]
}

BIKE_STEER_PID_PROFILES = {
    'vehicle.diamondback.century': [1.7, 10.0 / 8],
    'vehicle.gazelle.omafiets': [1.7, 10.0 / 8],
    'vehicle.harley-davidson.low_rider': [1.5, 10.0 / 9],
    'vehicle.bh.crossbike': [2.5, 10.0 / 8.0],
    'default': [2.0, 10.0 / 9]
}

# 转换 (K_p, T_u) -> (K_p, K_i, K_d)
for (k, v) in CAR_SPEED_PID_PROFILES.items():
    scale = 1.0
    CAR_SPEED_PID_PROFILES[k] = [0.8 * v[0] * scale, 0.0,  v[0] * v[1] / 10.0 * scale] # PD
    # CAR_SPEED_PID_PROFILES[k] = [v[0] / 5.0, v[0] * 2.0 / 5.0 / v[1], v[0] / 15.0 * v[1]] # no overshoot

for (k, v) in BIKE_SPEED_PID_PROFILES.items():
    BIKE_SPEED_PID_PROFILES[k] = [v[0] / 2.0, 0.0, 0.0] # p


for (k, v) in CAR_STEER_PID_PROFILES.items():
    scale = 0.9
    CAR_STEER_PID_PROFILES[k] = [v[0] / 2.0, 0.0, 0.0]


for (k, v) in BIKE_STEER_PID_PROFILES.items():
    scale = 0.9
    BIKE_STEER_PID_PROFILES[k] = [0.8 * v[0] * scale, 0.0,  v[0] * v[1] / 10.0 * scale ] # PD


# 根据车辆蓝图ID获取汽车速度PID配置
def get_car_speed_pid_profile(blueprint_id):
    result = CAR_SPEED_PID_PROFILES.get(blueprint_id)
    if result is not None:
        return result
    else:
        return CAR_SPEED_PID_PROFILES['default']

def get_bike_speed_pid_profile(blueprint_id):
    result = BIKE_SPEED_PID_PROFILES.get(blueprint_id)
    if result is not None:
        return result
    else:
        return BIKE_SPEED_PID_PROFILES['default']

def get_car_steer_pid_profile(blueprint_id):
    result = CAR_STEER_PID_PROFILES.get(blueprint_id)
    if result is not None:
        return result
    else:
        return CAR_STEER_PID_PROFILES['default']

def get_bike_steer_pid_profile(blueprint_id):
    result = BIKE_STEER_PID_PROFILES.get(blueprint_id)
    if result is not None:
        return result
    else:
        return BIKE_STEER_PID_PROFILES['default']

CAR_STEER_KP = 1.5 # 汽车转向比例增益
BIKE_STEER_KP = 1.0 # 自行车转向比例增益

Pyro4.config.SERIALIZERS_ACCEPTED.add('serpent') # 添加Pyro4序列化器
Pyro4.config.SERIALIZER = 'serpent' # 设置默认序列化器为serpent
# 注册carla.Vector2D类到Pyro4序列化器，用于网络传输
Pyro4.util.SerializerBase.register_class_to_dict(
        carla.Vector2D, 
        lambda o: { 
            '__class__': 'carla.Vector2D',
            'x': o.x,
            'y': o.y
        })
# 将字典反序列化为carla.Vector2D对象
Pyro4.util.SerializerBase.register_dict_to_class(
        'carla.Vector2D',
        lambda c, o: carla.Vector2D(o['x'], o['y']))
# 注册carla.SumoNetworkRoutePoint类到Pyro4序列化器，用于网络传输
Pyro4.util.SerializerBase.register_class_to_dict(
        carla.SumoNetworkRoutePoint,    
        lambda o: {     
            '__class__': 'carla.SumoNetworkRoutePoint', 
            'edge': o.edge, 
            'lane': o.lane, 
            'segment': o.segment,   
            'offset': o.offset  
        })

# 将字典反序列化为carla.SumoNetworkRoutePoint对象
def dict_to_sumo_network_route_point(c, o): 
    r = carla.SumoNetworkRoutePoint()   
    r.edge = str(o['edge'])  # 在Python2中，这是一个unicode字符串，所以使用str()进行转换
    r.lane = o['lane']  
    r.segment = o['segment']    
    r.offset = o['offset']  
    return r    

Pyro4.util.SerializerBase.register_dict_to_class(   
        'carla.SumoNetworkRoutePoint', dict_to_sumo_network_route_point)    
# 注册carla.SidewalkRoutePoint类到Pyro4序列化器，用于网络传输

Pyro4.util.SerializerBase.register_class_to_dict(   
        carla.SidewalkRoutePoint,   
        lambda o: {     
            '__class__': 'carla.SidewalkRoutePoint',    
            'polygon_id': o.polygon_id, 
            'segment_id': o.segment_id, 
            'offset': o.offset  
        })  
def dict_to_sidewalk_route_point(c, o): 
    r = carla.SidewalkRoutePoint()  
    r.polygon_id = o['polygon_id']  
    r.segment_id = o['segment_id']  
    r.offset = o['offset']
    return r    
Pyro4.util.SerializerBase.register_dict_to_class(   
        'carla.SidewalkRoutePoint', dict_to_sidewalk_route_point)


# 信息传递
''' ========== MESSAGE PASSING SERVICE ========== '''
@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class CrowdService():
    def __init__(self):
        self._simulation_bounds_min = None # 边界最小值
        self._simulation_bounds_max = None # 边界最大值
        self._simulation_bounds_lock = RLock() #边界锁 （可重入锁，用于在多线程中保护对象，防止多个线程同时修改同一资源，导致数据不一致）

        self._forbidden_bounds_min = None # 禁区最小值
        self._forbidden_bounds_max = None # 禁区最大值
        self._forbidden_bounds_lock = RLock() # 禁区锁

        self._spawn_car = False # 是否生成汽车
        self._new_cars = [] # 新汽车
        self._new_cars_lock = RLock() # 新汽车列表锁

        self._spawn_bike = False
        self._new_bikes = []
        self._new_bikes_lock = RLock()

        self._spawn_pedestrian = False
        self._new_pedestrians = []
        self._new_pedestrians_lock = RLock()

        self._control_velocities = [] # 控制速度列表
        self._control_velocities_lock = RLock() # 控制速度列表锁

        self._local_intentions = [] #本地意图列表
        self._local_intentions_lock = RLock() # 本地意图列表锁

        self._destroy_list = [] # 销毁列表
        self._destroy_list_lock = RLock() # 销毁列表锁

    @property
    def simulation_bounds(self):
        # 获取模拟边界
        self._simulation_bounds_lock.acquire()
        # 深拷贝操作，确保返回的 simulation_bounds_min 对象不会影响原始的 _simulation_bounds_min 对象。
        simulation_bounds_min = None if self._simulation_bounds_min is None else \
                carla.Vector2D(self._simulation_bounds_min.x, self._simulation_bounds_min.y)
        simulation_bounds_max = None if self._simulation_bounds_max is None else \
                carla.Vector2D(self._simulation_bounds_max.x, self._simulation_bounds_max.y)
        self._simulation_bounds_lock.release() # 释放锁，允许其他线程反访问_simulation_bounds_min 和 _simulation_bounds_max
        return (simulation_bounds_min, simulation_bounds_max)# 返回模拟边界元组

    @simulation_bounds.setter
    def simulation_bounds(self, bounds):
        # 设置模拟边界
        self._simulation_bounds_lock.acquire()
        self._simulation_bounds_min = bounds[0]
        self._simulation_bounds_max = bounds[1]
        self._simulation_bounds_lock.release() 
   

    @property
    def forbidden_bounds(self):
        # 获取禁止边界
        self._forbidden_bounds_lock.acquire()
        forbidden_bounds_min = None if self._forbidden_bounds_min is None else \
                carla.Vector2D(self._forbidden_bounds_min.x, self._forbidden_bounds_min.y)
        forbidden_bounds_max = None if self._forbidden_bounds_max is None else \
                carla.Vector2D(self._forbidden_bounds_max.x, self._forbidden_bounds_max.y)
        self._forbidden_bounds_lock.release()
        return (forbidden_bounds_min, forbidden_bounds_max)

    @forbidden_bounds.setter
    def forbidden_bounds(self, bounds):
        # 设置禁止边界
        self._forbidden_bounds_lock.acquire()
        self._forbidden_bounds_min = bounds[0]
        self._forbidden_bounds_max = bounds[1]
        self._forbidden_bounds_lock.release() 


    @property
    def spawn_car(self):
        # 获取生成汽车标志
        return self._spawn_car

    @spawn_car.setter
    def spawn_car(self, value):
        # 设置生成汽车标志
        self._spawn_car = value

    @property
    def new_cars(self):
        # 获取新汽车列表
        return self._new_cars

    @new_cars.setter
    def new_cars(self, cars):
        # 设置新汽车列表
        self._new_cars = cars
    
    def append_new_cars(self, info):
        # 添加新汽车信息到列表
        self._new_cars.append(info)

    def acquire_new_cars(self):
        # 获取新汽车锁
        self._new_cars_lock.acquire()

    def release_new_cars(self):
        # 释放新汽车锁
        try:
            self._new_cars_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()
   

    @property
    def spawn_bike(self):
        # 获取生成自行车标志
        return self._spawn_bike

    @spawn_bike.setter
    def spawn_bike(self, value):
        # 设置生成自行车标志
        self._spawn_bike = value

    @property
    def new_bikes(self):
        # 获取新自行车列表
        return self._new_bikes

    @new_bikes.setter
    def new_bikes(self, bikes):
        # 设置新自行车列表
        self._new_bikes = bikes
    
    def append_new_bikes(self, info):
        # 添加新自行车信息到列表
        self._new_bikes.append(info)
    
    def acquire_new_bikes(self):
        # 获取新自行车锁
        self._new_bikes_lock.acquire()

    def release_new_bikes(self):
        # 释放新自行车锁
        try:
            self._new_bikes_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()
    
    
    @property
    def spawn_pedestrian(self):
        # 获取生成行人标志
        return self._spawn_pedestrian

    @spawn_pedestrian.setter
    def spawn_pedestrian(self, value):
        # 设置生成行人标志
        self._spawn_pedestrian = value

    @property
    def new_pedestrians(self):
        # 获取新行人列表
        return self._new_pedestrians

    @new_pedestrians.setter
    def new_pedestrians(self, pedestrians):
        # 设置新行人列表
        self._new_pedestrians = pedestrians
    
    def append_new_pedestrians(self, info):
        # 添加新行人信息到列表
        self._new_pedestrians.append(info)
    
    def acquire_new_pedestrians(self):
        # 获取新行人锁
        self._new_pedestrians_lock.acquire()

    def release_new_pedestrians(self):
        # 释放新行人锁
        try:
            self._new_pedestrians_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()


    @property
    def control_velocities(self):
        # 获取控制速度列表
        return self._control_velocities

    @control_velocities.setter
    def control_velocities(self, velocities):
        # 设置控制速度列表
        self._control_velocities = velocities

    def acquire_control_velocities(self):
        # 获取控制速度锁
        self._control_velocities_lock.acquire()

    def release_control_velocities(self):
        # 释放控制速度锁
        try:
            self._control_velocities_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()

    @property
    def local_intentions(self):
        # 获取本地意图列表
        return self._local_intentions

    @local_intentions.setter
    def local_intentions(self, velocities):
        # 设置本地意图列表
        self._local_intentions = velocities

    def acquire_local_intentions(self):
        # 获取本地意图锁
        self._local_intentions_lock.acquire()

    def release_local_intentions(self):
        # 释放本地意图锁
        try:
            self._local_intentions_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()
   

    @property
    def destroy_list(self):
        # 获取销毁列表
        return self._destroy_list

    @destroy_list.setter
    def destroy_list(self, items):
        # 设置销毁列表
        self._destroy_list = items
    
    def append_destroy_list(self, item):
        # 添加销毁项到列表
        self._destroy_list.append(item)
    
    def extend_destroy_list(self, items):
        # 扩展销毁列表
        self._destroy_list.extend(items)

    def acquire_destroy_list(self):
        # 获取销毁列表锁
        self._destroy_list_lock.acquire()

    def release_destroy_list(self):
        # 释放销毁列表锁
        try:
            self._destroy_list_lock.release()
        except Exception as e:
            print(e)
            sys.stdout.flush()



''' ========== UTILITY FUNCTIONS AND CLASSES ========== '''

def get_signed_angle_diff(vector1, vector2):
    # 获取有符号角度差
    theta = math.atan2(vector1.y, vector1.x) - math.atan2(vector2.y, vector2.x)
    theta = np.rad2deg(theta)
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    return theta

def get_steer_angle_range(actor):
    # 获取转向角范围
    actor_physics_control = actor.get_physics_control()
    return (actor_physics_control.wheels[0].max_steer_angle + actor_physics_control.wheels[1].max_steer_angle) / 2

def get_position(actor):
    # 获取位置
    pos3d = actor.get_location()
    return carla.Vector2D(pos3d.x, pos3d.y)

def get_forward_direction(actor):
    # 获取前进方向
    forward = actor.get_transform().get_forward_vector()
    return carla.Vector2D(forward.x, forward.y)

def get_bounding_box(actor):
    # 获取包围盒
    return actor.bounding_box

def get_position_3d(actor):
    # 获取三维位置
    return actor.get_location()
# 获取代理的轴对齐边界框(AABB)
def get_aabb(actor):
    bbox = actor.bounding_box #获取代理的边界框
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor) # 获取代理的位置
    forward_vec = get_forward_direction(actor).make_unit_vector() # 获取代理的前进方向
    sideward_vec = forward_vec.rotate(np.deg2rad(90)) # 获取代理的侧向方向
    # 计算代理边界框的四个角
    corners = [loc - bbox.extent.x * forward_vec + bbox.extent.y * sideward_vec,
               loc + bbox.extent.x * forward_vec + bbox.extent.y * sideward_vec,
               loc + bbox.extent.x * forward_vec - bbox.extent.y * sideward_vec,
               loc - bbox.extent.x * forward_vec - bbox.extent.y * sideward_vec]
    # 返回代理的轴对齐边界框
    return carla.AABB2D(
        carla.Vector2D(
            min(v.x for v in corners),
            min(v.y for v in corners)),
        carla.Vector2D(
            max(v.x for v in corners),
            max(v.y for v in corners)))
# 获取代理的速度
def get_velocity(actor):
    v = actor.get_velocity()
    return carla.Vector2D(v.x, v.y)
# 获取代理的边界框的四个角   
def get_bounding_box_corners(actor):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(actor).make_unit_vector()
    sideward_vec = forward_vec.rotate(np.deg2rad(90))
    half_y_len = bbox.extent.y
    half_x_len = bbox.extent.x
    corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len * forward_vec - half_y_len * sideward_vec]
    return corners
# 获取车辆的边界框的四个角
def get_vehicle_bounding_box_corners(actor):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(actor).make_unit_vector()
    sideward_vec = forward_vec.rotate(np.deg2rad(90))
    half_y_len = bbox.extent.y + 0.3
    half_x_len_forward = bbox.extent.x + 1.0
    half_x_len_backward = bbox.extent.x + 0.1
    corners = [loc - half_x_len_backward * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len_forward * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len_forward * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len_backward * forward_vec - half_y_len * sideward_vec]
    return corners
# 获取行人的边界框的四个角
def get_pedestrian_bounding_box_corners(actor):
    bbox = actor.bounding_box
    loc = carla.Vector2D(bbox.location.x, bbox.location.y) + get_position(actor)
    forward_vec = get_forward_direction(actor).make_unit_vector()
    sideward_vec = forward_vec.rotate(np.deg2rad(90))
    # Hardcoded values for pedestrians.
    half_y_len = 0.25
    half_x_len = 0.25
    corners = [loc - half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec + half_y_len * sideward_vec,
               loc + half_x_len * forward_vec - half_y_len * sideward_vec,
               loc - half_x_len * forward_vec - half_y_len * sideward_vec]
    return corners
#获取车道约束，确定左右两条车道是否受到人行道的限制。
def get_lane_constraints(sidewalk, position, forward_vec):
    left_line_end = position + (1.5 + 2.0 + 0.8) * ((forward_vec.rotate(np.deg2rad(-90))).make_unit_vector())
    right_line_end = position + (1.5 + 2.0 + 0.8) * ((forward_vec.rotate(np.deg2rad(90))).make_unit_vector())
    left_lane_constrained_by_sidewalk = sidewalk.intersects(carla.Segment2D(position, left_line_end))
    right_lane_constrained_by_sidewalk = sidewalk.intersects(carla.Segment2D(position, right_line_end))
    # 返回一个包含两个布尔值的元组，这两个布尔值分别表示左右两条车道是否受到人行道的限制。
    return left_lane_constrained_by_sidewalk, right_lane_constrained_by_sidewalk
# 类型判断
def is_car(actor):
    return isinstance(actor, carla.Vehicle) and int(actor.attributes['number_of_wheels']) > 2

def is_bike(actor):
    return isinstance(actor, carla.Vehicle) and int(actor.attributes['number_of_wheels']) == 2

def is_pedestrian(actor):
    return isinstance(actor, carla.Walker)

class Graph:
    def __init__(self):
        self.edges = {}
    
    def add_edge(self, start, end, cost):
        if start not in self.edges:
            self.edges[start] = []
        self.edges[start].append((end, cost))

# 车辆代理路径
class SumoNetworkAgentPath:
    def __init__(self, route_points, min_points, interval, destination_pos=None):
        # 初始化路径点，最小点数和间隔，设置为实例变量
        self.route_points = route_points 
        self.min_points = min_points
        self.interval = interval
        self.destination_pos = None # 目的地位置

    @staticmethod
    def rand_path(sumo_network, min_points, destination,interval, segment_map, rng=random):
        spawn_point = None
        route_paths = None
        while not spawn_point or len(route_paths) < 1:
            spawn_point = segment_map.rand_point()
            spawn_point = sumo_network.get_nearest_route_point(spawn_point)

            # 使用Dijkstra算法找到最优路径
            route_paths = SumoNetworkAgentPath.find_best_path(sumo_network, spawn_point, destination, min_points, interval)
        
        return SumoNetworkAgentPath(rng.choice(route_paths), min_points, interval)
    
    @staticmethod
    # 计算方向权重
    def calc_direction_weight(sumo_network, current_pos, next_pos, destination_pos):
        """
        计算从当前位置到下一个位置的方向权重，相对于当前位置到目的地位置的方向。
        :param current_pos: 当前位置的坐标,包含x和y的对象,
        :param next_pos: 下一个位置的坐标,格式同current_pos。
        :param destination_pos: 目的地位置的坐标,格式同current_pos。
        :return: 方向权重，范围[0, 1]，值越大表示方向越接近目的地。
        """
        if isinstance(current_pos, carla.SumoNetworkRoutePoint):
            current_pos = sumo_network.get_route_point_position(current_pos)
        if isinstance(next_pos, carla.SumoNetworkRoutePoint):
            next_pos = sumo_network.get_route_point_position(next_pos)
        if isinstance(destination_pos, carla.SumoNetworkRoutePoint):
            destination_pos = sumo_network.get_route_point_position(destination_pos)
        try:
            # 计算当前位置到目的地的方向
            current_yaw = np.rad2deg(math.atan2(next_pos.y - current_pos.y, next_pos.x - current_pos.x))
            destination_yaw = np.rad2deg(math.atan2(destination_pos.y - current_pos.y, destination_pos.x - current_pos.x))
            
            # 计算偏航角差
            yaw_diff = abs(current_yaw - destination_yaw)
            # 确保偏航角差在[0, 180]范围内
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            
            # 计算方向权重，这里我们使用余弦值来表示方向的接近程度
            direction_weight = math.cos(math.radians(yaw_diff))
                
            return direction_weight
        except IndexError:
            return 1.0  # 给默认值1.0
    @staticmethod
    def calc_safety_weight(sumo_network,next_vertex, occupied_vertices, selected_vertices):
        """
        根据下一个位置的安全状况计算安全权值，结合了场景中已有车辆和行人的占用信息。
        
        :param next_pos: 考虑移动到的下一个位置,格式为carla.Vector2D。
        :param occupied_positions: 已经被占用的位置列表。
        :param selected_positions: 已经被选择的位置列表。
        :如果下一个位置在已经被占用的位置列表中,返回0.5,表示预警位置。
        :如果下一个位置在已经被选择的位置列表中,返回0.1,表示危险位置。
        :如果下一个位置不在以上两个列表中,返回0.9,表示安全位置。
        :return: 下一个位置的安全权值。
        """
        if isinstance(next_vertex, carla.SumoNetworkRoutePoint):
           next_vertex = sumo_network.get_route_point_position(next_vertex)
        is_occupied = any(next_vertex.x == v.x and next_vertex.y == v.y for v in occupied_vertices)
        is_selected = any(next_vertex.x == v.x and next_vertex.y == v.y for v in selected_vertices)

        if is_occupied:
            return 0.5
        elif is_selected:
            return 0.1
        else:
            return 0.9
    @staticmethod   
    # 计算转向角度
    def calculate_angle(current_pos, next_pos):
        """
        计算两点形成的向量与正Y轴的夹角,用于判断转向。
        :param current_pos: 当前位置，格式为(x, y)。
        :param next_pos: 下一个位置，格式为(x, y)。
        :return: 与正Y轴的夹角,范围[0, 360)。
        """
        dx = next_pos.x - current_pos.x
        dy = next_pos.y - current_pos.y
        angle_radians = math.atan2(dy, dx)  # 计算角度的弧度值
        angle_degrees = math.degrees(angle_radians)  # 转换为度
        # 将角度转换为[0, 360)范围
        angle = (angle_degrees + 360) % 360
        return angle
    @staticmethod
    # 计算优先级权重，根据车辆当前位置与下一个位置之间的角度，判断出直行左转还是右转
    def calc_priority_weight(sumo_network, current_pos, next_pos):
        if isinstance(current_pos, carla.SumoNetworkRoutePoint):
            current_pos = sumo_network.get_route_point_position(current_pos)
        if isinstance(next_pos, carla.SumoNetworkRoutePoint):
            next_pos = sumo_network.get_route_point_position(next_pos)
        angle = SumoNetworkAgentPath.calculate_angle(current_pos, next_pos)
        # 假设北方（正Y轴方向）为0度，东方为90度，南方为180度，西方为270度
        # 直行（考虑一定的容差，例如正北或正南方向±45度范围内为直行）
        if 45 < angle < 135 or 225 < angle < 315:
            # 直行
            return 0.8
        elif 0 <= angle <= 45 or 315 < angle < 360 or 135 < angle < 225:
            # 左转
            return 0.6
        else:
            # 右转
            return 0.4
    @staticmethod    
    def choose_destination(sumo_network, current_node, min_points, interval):
        dest = []
        current_node = sumo_network.get_nearest_route_point(current_node)
        destination_pos = sumo_network.get_next_route_paths(current_node, min_points - 1, interval)
        """print("destination_pos",destination_pos)
        print("*******")"""
        for list in destination_pos:
            for point in list:
                dest.append(point)
        """print("dest",dest)"""
        destination_pos = dest[-1]
        """print("destination_pos",destination_pos)"""

        return destination_pos
    @staticmethod
    # 计算路径权重
    def calculate_weight(sumo_network, current_pos, next_pos, destination_pos):
        """
        计算路径权重，包括方向权重、安全权重和优先级权重。
        :param current_pos: 当前位置的坐标,包含x和y的对象。
        :param next_pos: 下一个位置的坐标,格式同current_pos。
        :param destination_pos: 目的地位置的坐标,格式同current_pos。
        :return: 路径权重，范围[0, 1]，值越大表示路径越优。
        """
        direction_weight = SumoNetworkAgentPath.calc_direction_weight(sumo_network, current_pos, next_pos, destination_pos)
        safety_weight = SumoNetworkAgentPath.calc_safety_weight(sumo_network, next_pos, [], [])
        priority_weight = SumoNetworkAgentPath.calc_priority_weight(sumo_network, current_pos, next_pos)
        # 计算路径权重
        weight = direction_weight * safety_weight * priority_weight 
        print({'current_pos':current_pos,'next_pos':next_pos,'destination_pos':destination_pos,'direction_weight':direction_weight,'safety_weight':safety_weight,'priority_weight':priority_weight,'weight':weight})
        return weight
    
    @staticmethod
    def find_best_path(sumo_network, start_point,destination, min_points, interval):
        # 使用简化的Dijkstra算法找到最优路径
        visited = set()
        paths = [[start_point]]
        best_paths = []

        while paths:
            path = paths.pop(0)
            current_point = path[-1]

            if len(path) >= min_points:
                best_paths.append(path)
                continue

            visited.add(current_point)

            next_points = sumo_network.get_next_route_points(current_point, interval)
            for next_point in next_points:
                if next_point not in visited:
                    new_path = list(path)
                    new_path.append(next_point)
                    paths.append(new_path)
            
            # 根据路径权值排序
            paths.sort(key=lambda x: sum(SumoNetworkAgentPath.calculate_weight(sumo_network, x[i], x[i+1],destination) for i in range(len(x)-1)))
        
        return best_paths
    @staticmethod
    def get_path_accumulated_weight(sumo_network, route_points, occupied_vertices, selected_vertices):
        accumulated_weight = 0
        for i in range(len(route_points) - 1):
            weight = SumoNetworkAgentPath.calculate_weight(sumo_network, route_points[i], route_points[i+1], route_points[-1])
            accumulated_weight += weight
        return accumulated_weight    

    # 调整路径长度，在路径的末尾添加新的路径点，如果无法添加新的路径点，则返回False，（可能到达网络的边界）                       
    def resize(self, sumo_network, rng=random):
        while len(self.route_points) < self.min_points:#如果路径长度小于最小点数
            next_points = sumo_network.get_next_route_points(self.route_points[-1], self.interval)# 获取下一个路径点
            if len(next_points) == 0:# 如果没有下一个路径点
                return False
            self.route_points.append(rng.choice(next_points)) # 添加到路径点列表中
        return True
    
    
        
    # 获取与指定位置最近的路径点的偏移量
    def get_min_offset(self, sumo_network, position):
        min_offset = None
        for i in range(int(len(self.route_points) / 2)):# 遍历路径点
            route_point = self.route_points[i]# 获取路径点
            offset = position - sumo_network.get_route_point_position(route_point)# 计算偏移量
            offset = offset.length() # 计算长度destination_pos
        return min_offset
    
    # 切割路径，删除路径中与指定位置最近的路径点之前的所有路径点，使对象从离指定位置最近的路径点开始
    def cut(self, sumo_network, position):
        cut_index = 0 # 切割索引
        min_offset = None 
        min_offset_index = None # 最小偏移量索引
        for i in range(int(len(self.route_points) / 2)): # 遍历路径点
            route_point = self.route_points[i] # 获取路径点
            offset = position - sumo_network.get_route_point_position(route_point)# 计算偏移量
            offset = offset.length() 
            if min_offset == None or offset < min_offset: # 如果偏移量为空或者偏移量小于最小偏移量
                min_offset = offset # 更新最小偏移量
                min_offset_index = i # 更新最小偏移量索引
            if offset <= 1.0: # 如果偏移量小于等于1.0
                cut_index = i + 1 # 更新切割索引

        # Invalid path because too far away.
        if min_offset > 1.0: # 如果最小偏移量大于1.0
            self.route_points = self.route_points[min_offset_index:] # 如果大于1，则路径从最近的点开始
        else: 
            self.route_points = self.route_points[cut_index:] # 否则路径从第一个偏移量小于或等于1.0的点开始

    # 获取路径中给定索引点的位置
    def get_position(self, sumo_network, index=0):
        return sumo_network.get_route_point_position(self.route_points[index])
    
    # 获取路径中给定索引点的偏航角
    def get_yaw(self, sumo_network, index=0):
        pos = sumo_network.get_route_point_position(self.route_points[index]) # 获取路径点的位置
        next_pos = sumo_network.get_route_point_position(self.route_points[index + 1]) # 获取下一个路径点的位置
        return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x)) # 计算偏航角
    

 
# 人行道上的代理路径
class SidewalkAgentPath:
    def __init__(self, route_points, route_orientations, min_points, interval):
        self.min_points = min_points # 最小点数
        self.interval = interval # 间隔
        self.route_points = route_points # 路径点
        self.route_orientations = route_orientations # 路径方向

    @staticmethod
    # 静态方法，随机生成人行道上的代理路径
    # sidewalk:人行道，min_points:最小点数，interval:间隔，cross_probability:横穿概率，segment_map:段地图，rng:随机数生成器
    def rand_path(sidewalk, min_points, interval, cross_probability, segment_map, rng=None):
        if rng is None: # 如果随机数为空
            rng = random # 使用默认的随机数生成器
    
        spawn_point = sidewalk.get_nearest_route_point(segment_map.rand_point()) # 获取最近的路径点

        path = SidewalkAgentPath([spawn_point], [rng.choice([True, False])], min_points, interval) # 创建一个新的人行道代理路径
        path.resize(sidewalk, cross_probability) # 调整路径长度
        return path 
    # 调整路径长度，根据当前路径的最后一个方向，获取并添加新的路径点，以及对应的方向。
    def resize(self, sidewalk, cross_probability, rng=None):
        if rng is None: 
            rng = random 

        while len(self.route_points) < self.min_points:# 如果路径长度小于最小点数
            if rng.random() <= cross_probability: # 如果随机数小于等于横穿概率
                adjacent_route_point = sidewalk.get_adjacent_route_point(self.route_points[-1], 50.0) # 尝试获取相邻的路径点
                if adjacent_route_point is not None:
                    self.route_points.append(adjacent_route_point) # 将获取到的相邻路径点添加到路径点列表中
                    self.route_orientations.append(rng.randint(0, 1) == 1) # 随机选择一个方向
                    continue # 继续循环

            if self.route_orientations[-1]: # 如果路径方向为True
                self.route_points.append(
                        sidewalk.get_next_route_point(self.route_points[-1], self.interval)) # 获取下一个路径点
                self.route_orientations.append(True) # 添加路径方向
            else:
                self.route_points.append(
                        sidewalk.get_previous_route_point(self.route_points[-1], self.interval)) # 获取上一个路径点
                self.route_orientations.append(False) # 添加路径方向

        return True
    # 根据当前位置剪切路径点 (同Sumo)
    def cut(self, sidewalk, position):
        cut_index = 0
        min_offset = None
        min_offset_index = None
        for i in range(int(len(self.route_points) / 2)):
            route_point = self.route_points[i]
            offset = position - sidewalk.get_route_point_position(route_point)
            offset = offset.length()
            if min_offset is None or offset < min_offset:
                min_offset = offset
                min_offset_index = i
            if offset <= 1.0:
                cut_index = i + 1
        
        # Invalid path because too far away.
        if min_offset > 1.0:
            self.route_points = self.route_points[min_offset_index:]
            self.route_orientations = self.route_orientations[min_offset_index:]
        else:
            self.route_points = self.route_points[cut_index:]
            self.route_orientations = self.route_orientations[cut_index:]
    # 获取路径中给定索引点的位置
    def get_position(self, sidewalk, index=0):
        return sidewalk.get_route_point_position(self.route_points[index])
    # 获取路径中给定索引点的偏航角
    def get_yaw(self, sidewalk, index=0):
        pos = sidewalk.get_route_point_position(self.route_points[index]) # 获取路径点的位置
        next_pos = sidewalk.get_route_point_position(self.route_points[index + 1]) # 获取下一个路径点的位置
        return np.rad2deg(math.atan2(next_pos.y - pos.y, next_pos.x - pos.x)) # 计算偏航角
    
# 类，代表一个行为主体
class Agent(object):
    def __init__(self, actor, type_tag, path, preferred_speed, steer_angle_range=0.0, rand=0): 
        # 初始化行为主体
        self.actor = actor # 代理的演员（车辆或者行人）
        self.type_tag = type_tag # 代理的类型标签（例如“Car”、“Bike”、“Pedestrian”）
        self.path = path # 代理应该遵循的路径
        self.preferred_speed = preferred_speed # 代理的首选速度
        self.stuck_time = None # 用于记录代理被卡住的时间
        self.control_velocity = carla.Vector2D(0, 0) # 控制代理的速度
        self.steer_angle_range = steer_angle_range # 控制代理的转向角范围
        self.behavior_type = self.rand_agent_behavior_type(rand) # 随机确定代理的行为类型
        self.occupied_vertex = None # 代理所占用的路径点
        self.selected_vertex = None # 代理选择的路径点
        self.other_vehicles = [] # 代理周围的其他车辆
        self.path_data = [] # 
    def rand_agent_behavior_type(self, prob): #接收一个prob概率值，根据概率随机决定代理的行为类型
        prob_gamma_agent = 1.0 # GAMMA行为主体的概率
        prob_simplified_gamma_agent = 0.0 # 简化版GAMMA行为主体的概率
        prob_ttc_agent = 0.0 # 基于时间到碰撞的行为主体的概率

        if prob <= prob_gamma_agent: # 根据prob的值返回行为类型
            return carla.AgentBehaviorType.Gamma # GAMMA行为主体
        elif prob <= prob_gamma_agent + prob_simplified_gamma_agent: 
            return carla.AgentBehaviorType.SimplifiedGamma # 简化GAMMA行为主体
        else:
            return -1 #表示未定义的行为类型

# 类，用于初始化和管理与模拟环境相关的各种参数和对象
class Context(object):
    def __init__(self, args):
        self.args = args # 传入参数
        self.rng = random.Random(args.seed) # 随机数生成器
        #从文件中读取模拟边界并创建模拟边界的占用地图
        with (DATA_PATH/'{}.sim_bounds'.format(args.dataset)).open('r') as f: # 从文件中读取模拟边界
            self.bounds_min = carla.Vector2D(*[float(v) for v in f.readline().split(',')]) 
            self.bounds_max = carla.Vector2D(*[float(v) for v in f.readline().split(',')])
            self.bounds_occupancy = carla.OccupancyMap(self.bounds_min, self.bounds_max)
        
        self.forbidden_bounds_min = None # 禁止边界的最小值
        self.forbidden_bounds_max = None # 禁止边界的最大值
        self.forbidden_bounds_occupancy = None # 禁止边界的占用地图
        # 加载SUMO网络，并创建了一个段地图
        self.sumo_network = carla.SumoNetwork.load(str(DATA_PATH/'{}.net.xml'.format(args.dataset)))
        self.sumo_network_segments = self.sumo_network.create_segment_map()
        # 找出段地图中的模拟边界的交集，保存在sumo_network_spawn_segments中用于生成车辆
        self.sumo_network_spawn_segments = self.sumo_network_segments.intersection(carla.OccupancyMap(self.bounds_min, self.bounds_max))
        self.sumo_network_spawn_segments.seed_rand(self.rng.getrandbits(32))
        self.sumo_network_occupancy = carla.OccupancyMap.load(str(DATA_PATH/'{}.network.wkt'.format(args.dataset)))
        # 创建人行道
        # 从文件中读取人行道并创建人行道的段地图
        # 找出人行道段地图中的模拟边界的交集，保存在sidewalk_spawn_segments中用于生成行人
        self.sidewalk = self.sumo_network_occupancy.create_sidewalk(1.5)
        self.sidewalk_segments = self.sidewalk.create_segment_map()
        self.sidewalk_spawn_segments = self.sidewalk_segments.intersection(carla.OccupancyMap(self.bounds_min, self.bounds_max))
        self.sidewalk_spawn_segments.seed_rand(self.rng.getrandbits(32))
        self.sidewalk_occupancy = carla.OccupancyMap.load(str(DATA_PATH/'{}.sidewalk.wkt'.format(args.dataset)))
        # 设置CARLA客户端，超时时间为10秒
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(10.0)
        # 获取CARLA世界对象，并创建了一Pyro4代理来连接到远程的crowd_service 人群服务
        self.world = self.client.get_world()
        # 获取carla路径点
        self.points = carla.SumoNetworkRoutePoint
        self.occupied_vertex = [] # 占用的路径点
        self.selected_vertex = [] # 选择的路径点
        self.crowd_service = Pyro4.Proxy('PYRO:crowdservice.warehouse@localhost:{}'.format(args.pyroport))
        # 获取CARLA世界中的所有车辆和行人的蓝图
        self.pedestrian_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*') # 行人蓝图
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')# 车辆蓝图
        self.car_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4] #四轮车蓝图
        self.car_blueprints = [x for x in self.car_blueprints if x.id not in ['vehicle.bmw.isetta', 'vehicle.tesla.cybertruck']] # This dude moves too slow.
        self.bike_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 2] # 两轮车蓝图
# 类，用于记录模拟环境中的统计信息
class Statistics(object):
    def __init__(self, log_file):
        self.start_time = None #统计开始的时间

        self.total_num_cars = 0 # 总汽车数
        self.total_num_bikes = 0 # 总自行车数
        self.total_num_pedestrians = 0 # 总行人数 

        self.stuck_num_cars = 0 # 被卡住的汽车数 
        self.stuck_num_bikes = 0 # 被卡住的自行车数 
        self.stuck_num_pedestrians = 0 # 被卡住的行人数
 
        self.avg_speed_cars = 0 # 平均汽车速度
        self.avg_speed_bikes = 0 # 平均自行车速度
        self.avg_speed_pedestrians = 0 # 平均行人速度

        self.log_file = log_file # 用于记录统计信息的日志文件

    def write(self):
        # 将统计信息写入日志文件
        self.log_file.write('{} {} {} {} {} {} {} {} {} {}\n'.format(
            time.time() - self.start_time,# 从统计开始到现在的时间
            self.total_num_cars, 
            self.total_num_bikes, 
            self.total_num_pedestrians,
            self.stuck_num_cars, 
            self.stuck_num_bikes, 
            self.stuck_num_pedestrians,
            self.avg_speed_cars,
            self.avg_speed_bikes,
            self.avg_speed_pedestrians))
        self.log_file.flush() #清空文件缓冲区，确保所有的数据都被写入文件
        os.fsync(self.log_file) #确保文件的改动被同步到磁盘


''' ========== MAIN LOGIC FUNCTIONS ========== '''
def update_vehicle_state(vehicle, recorder):
    # 假设这里是更新车辆状态的代码
    # ...
    # 记录数据
    recorder.add_record(
        position=str(vehicle.get_location()), 
        velocity=str(vehicle.get_velocity()),
        acceleration=str(vehicle.get_acceleration()),
        steer=str(vehicle.get_control().steer))

# 在模拟环境中生成新的车辆、自行车和行人
def do_spawn(c):
    # 生成新的车辆、自行车和行人
    c.crowd_service.acquire_new_cars()
    spawn_car = c.crowd_service.spawn_car
    c.crowd_service.release_new_cars() # 释放新车辆锁
    
    c.crowd_service.acquire_new_bikes()
    spawn_bike = c.crowd_service.spawn_bike
    c.crowd_service.release_new_bikes()

    occupied_vertices = c.occupied_vertex
    selected_vertices = c.selected_vertex

    c.crowd_service.acquire_new_pedestrians()
    spawn_pedestrian = c.crowd_service.spawn_pedestrian
    c.crowd_service.release_new_pedestrians()
    # 如果没有车辆、自行车和行人需要生成，则直接返回
    if not spawn_car and not spawn_bike and not spawn_pedestrian:
        return

    # Find car spawn point.
    if spawn_car:
        # 根据当前场景中的障碍物创建一个占用地图
        aabb_occupancy = carla.OccupancyMap() if c.forbidden_bounds_occupancy is None else c.forbidden_bounds_occupancy
        # 遍历场景中的所有车辆和行人，获取他们的轴对齐边界框，并将其合并到占用地图中
        for actor in c.world.get_actors():
            # 如果是车辆或者行人
            if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
                aabb = get_aabb(actor)
                aabb_occupancy = aabb_occupancy.union(carla.OccupancyMap(
                    carla.Vector2D(aabb.bounds_min.x - c.args.clearance_car, aabb.bounds_min.y - c.args.clearance_car), 
                    carla.Vector2D(aabb.bounds_max.x + c.args.clearance_car, aabb.bounds_max.y + c.args.clearance_car)))
        # 尝试生成车辆直到成功或者达到最大尝试次数
        for _ in range(SPAWN_DESTROY_REPETITIONS): #SPAWN_DESTROY_REPETITIONS为循环次数

            spawn_segments = c.sumo_network_spawn_segments.difference(aabb_occupancy)
            """print("spawn_segments",spawn_segments)"""
            # 如果没有可用的生成点，则继续下一次循环
            if spawn_segments.is_empty:
                continue
            #用随机数生成器随机选择一个生成的位置生成新的车辆
            spawn_segments.seed_rand(c.rng.getrandbits(32))
            """print("dudu?",spawn_segments.seed_rand(c.rng.getrandbits(32)))"""
            current_pos = spawn_segments.rand_point()            
            # 生成车辆的目的地
            destination = SumoNetworkAgentPath.choose_destination(c.sumo_network, current_pos, PATH_MIN_POINTS, PATH_INTERVAL)
            # 从SUMO网络中用rand_path生成一条路径
            path = SumoNetworkAgentPath.rand_path(c.sumo_network,PATH_MIN_POINTS,destination, PATH_INTERVAL,spawn_segments, rng=c.rng)
            
            # 获取路径的起始位置 0代表索引起始位置
            position = path.get_position(c.sumo_network, 0)
            trans = carla.Transform()
            trans.location.x = position.x
            trans.location.y = position.y
            trans.location.z = 0.2
            trans.rotation.yaw = path.get_yaw(c.sumo_network, 0)
            
            actor = c.world.try_spawn_actor(c.rng.choice(c.car_blueprints), trans)
            
            if actor:
                # 设置车辆的碰撞检测，等待车辆更新位置和边界，以及碰撞应用
                actor.set_collision_enabled(c.args.collision)
                c.world.wait_for_tick(1.0)  # For actor to update pos and bounds, and for collision to apply.
                # 获取新车辆的锁，将新车辆添加到新车辆列表中，其中包括车辆的id、路径点和转向角范围
                c.crowd_service.acquire_new_cars()
                c.crowd_service.append_new_cars((
                    actor.id, 
                    [p for p in path.route_points], # Convert to python list.
                    get_steer_angle_range(actor)))
                # 释放新车辆的锁
                c.crowd_service.release_new_cars()
                # 获取车辆的轴对齐边界框，并将其合并到占用地图中
                aabb = get_aabb(actor)
                aabb_occupancy = aabb_occupancy.union(carla.OccupancyMap(
                    carla.Vector2D(aabb.bounds_min.x - c.args.clearance_car, aabb.bounds_min.y - c.args.clearance_car), 
                    carla.Vector2D(aabb.bounds_max.x + c.args.clearance_car, aabb.bounds_max.y + c.args.clearance_car)))
             
            
    # Find bike spawn point.
    if spawn_bike:
        aabb_occupancy = carla.OccupancyMap() if c.forbidden_bounds_occupancy is None else c.forbidden_bounds_occupancy
        for actor in c.world.get_actors():
            if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
                aabb = get_aabb(actor)
                aabb_occupancy = aabb_occupancy.union(carla.OccupancyMap(
                    carla.Vector2D(aabb.bounds_min.x - c.args.clearance_bike, aabb.bounds_min.y - c.args.clearance_bike), 
                    carla.Vector2D(aabb.bounds_max.x + c.args.clearance_bike, aabb.bounds_max.y + c.args.clearance_bike)))
        
        for _ in range(SPAWN_DESTROY_REPETITIONS):
            spawn_segments = c.sumo_network_spawn_segments.difference(aabb_occupancy)
            if spawn_segments.is_empty:
                continue
            spawn_segments.seed_rand(c.rng.getrandbits(32))
            
            path = SumoNetworkAgentPath.rand_path(c.sumo_network, PATH_MIN_POINTS, PATH_INTERVAL, spawn_segments, rng=c.rng)
            position = path.get_position(c.sumo_network, 0)
            trans = carla.Transform()
            trans.location.x = position.x
            trans.location.y = position.y
            trans.location.z = 0.2
            trans.rotation.yaw = path.get_yaw(c.sumo_network, 0)

            actor = c.world.try_spawn_actor(c.rng.choice(c.bike_blueprints), trans)
            if actor:
                actor.set_collision_enabled(c.args.collision)
                c.world.wait_for_tick(1.0)  # For actor to update pos and bounds, and for collision to apply.
                c.crowd_service.acquire_new_bikes()
                c.crowd_service.append_new_bikes((
                    actor.id, 
                    [p for p in path.route_points], # Convert to python list.
                    get_steer_angle_range(actor)))
                c.crowd_service.release_new_bikes()
                aabb = get_aabb(actor)
                aabb_occupancy = aabb_occupancy.union(carla.OccupancyMap(
                    carla.Vector2D(aabb.bounds_min.x - c.args.clearance_bike, aabb.bounds_min.y - c.args.clearance_bike), 
                    carla.Vector2D(aabb.bounds_max.x + c.args.clearance_bike, aabb.bounds_max.y + c.args.clearance_bike)))


    if spawn_pedestrian:
        aabb_occupancy = carla.OccupancyMap() if c.forbidden_bounds_occupancy is None else c.forbidden_bounds_occupancy
        for actor in c.world.get_actors():
            if isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker):
                aabb = get_aabb(actor)
                aabb_occupancy = aabb_occupancy.union(carla.OccupancyMap(
                    carla.Vector2D(aabb.bounds_min.x - c.args.clearance_pedestrian, aabb.bounds_min.y - c.args.clearance_pedestrian), 
                    carla.Vector2D(aabb.bounds_max.x + c.args.clearance_pedestrian, aabb.bounds_max.y + c.args.clearance_pedestrian)))
        
        for _ in range(SPAWN_DESTROY_REPETITIONS):
            spawn_segments = c.sidewalk_spawn_segments.difference(aabb_occupancy)
            if spawn_segments.is_empty:
                continue
            spawn_segments.seed_rand(c.rng.getrandbits(32))

            path = SidewalkAgentPath.rand_path(c.sidewalk, PATH_MIN_POINTS, PATH_INTERVAL, c.args.cross_probability, c.sidewalk_spawn_segments, c.rng)
            position = path.get_position(c.sidewalk, 0)
            trans = carla.Transform()
            trans.location.x = position.x
            trans.location.y = position.y
            trans.location.z = 0.5
            trans.rotation.yaw = path.get_yaw(c.sidewalk, 0)
            actor = c.world.try_spawn_actor(c.rng.choice(c.pedestrian_blueprints), trans)
            if actor:
                actor.set_collision_enabled(c.args.collision)
                c.world.wait_for_tick(1.0)  # For actor to update pos and bounds, and for collision to apply.
                c.crowd_service.acquire_new_pedestrians()
                c.crowd_service.append_new_pedestrians((
                    actor.id, 
                    [p for p in path.route_points], # Convert to python list.
                    path.route_orientations))
                c.crowd_service.release_new_pedestrians()
                aabb = get_aabb(actor)
                aabb_occupancy = aabb_occupancy.union(carla.OccupancyMap(
                    carla.Vector2D(aabb.bounds_min.x - c.args.clearance_pedestrian, aabb.bounds_min.y - c.args.clearance_pedestrian), 
                    carla.Vector2D(aabb.bounds_max.x + c.args.clearance_pedestrian, aabb.bounds_max.y + c.args.clearance_pedestrian)))

# 在模拟环境中销毁车辆、自行车和行人
def do_destroy(c):
    # 获取销毁列表的锁，获取销毁列表并清空销毁列表
    c.crowd_service.acquire_destroy_list()
    destroy_list = c.crowd_service.destroy_list
    c.crowd_service.destroy_list = []
    # 释放销毁列表的锁，让其他线程也可以访问销毁列表
    c.crowd_service.release_destroy_list()
    # 创建销毁代理的命令列表，传入实体的id
    commands = [carla.command.DestroyActor(x) for x in destroy_list]
    #执行销毁命令并等待一个tick，确保销毁命令生效
    c.client.apply_batch_sync(commands)
    c.world.wait_for_tick(1.0)

# 从服务中获取新生成的代理信息，并将其添加到代理列表中
def pull_new_agents(c, car_agents, bike_agents, pedestrian_agents, statistics):
    
    new_car_agents = []
    new_bike_agents = []
    new_pedestrian_agents = []
    # 获取新的车辆代理
    c.crowd_service.acquire_new_cars()
    for (actor_id, route_points, steer_angle_range) in c.crowd_service.new_cars:
        path = SumoNetworkAgentPath(route_points, PATH_MIN_POINTS, PATH_INTERVAL) # 车辆路径
        new_car_agents.append(Agent(
            c.world.get_actor(actor_id), 'Car', path, c.args.speed_car + c.rng.uniform(-0.5, 0.5),
            steer_angle_range, rand=c.rng.uniform(0.0, 1.0)))
    c.crowd_service.new_cars = []
    c.crowd_service.spawn_car = len(car_agents) < c.args.num_car # 判断现有车辆代理数量是否小于最大车辆代理数量 
    c.crowd_service.release_new_cars() # 释放新的车辆代理
    
    c.crowd_service.acquire_new_bikes()
    for (actor_id, route_points, steer_angle_range) in c.crowd_service.new_bikes:
        path = SumoNetworkAgentPath(route_points, PATH_MIN_POINTS, PATH_INTERVAL)
        new_bike_agents.append(Agent(
            c.world.get_actor(actor_id), 'Bicycle', path, c.args.speed_bike + c.rng.uniform(-0.5, 0.5),
            steer_angle_range, rand=c.rng.uniform(0.0, 1.0)))
    c.crowd_service.new_bikes = []
    c.crowd_service.spawn_bike = len(bike_agents) < c.args.num_bike
    c.crowd_service.release_new_bikes()
    
    c.crowd_service.acquire_new_pedestrians()
    for (actor_id, route_points, route_orientations) in c.crowd_service.new_pedestrians:
        path = SidewalkAgentPath(route_points, route_orientations, PATH_MIN_POINTS, PATH_INTERVAL)
        path.resize(c.sidewalk, c.args.cross_probability)
        new_pedestrian_agents.append(Agent(
            c.world.get_actor(actor_id), 'People', path, c.args.speed_pedestrian + c.rng.uniform(-0.5, 0.5),
            rand=c.rng.uniform(0.0, 1.0)))
    c.crowd_service.new_pedestrians = []
    c.crowd_service.spawn_pedestrian = len(pedestrian_agents) < c.args.num_pedestrian
    c.crowd_service.release_new_pedestrians()
    # 更新统计数据
    statistics.total_num_cars += len(new_car_agents)
    statistics.total_num_bikes += len(new_bike_agents)
    statistics.total_num_pedestrians += len(new_pedestrian_agents)
    # 返回更新后的代理列表和统计数据
    return (car_agents + new_car_agents, bike_agents + new_bike_agents, pedestrian_agents + new_pedestrian_agents, statistics)

# 更新代理列表，移除需要销毁的代理：离开模拟区域或卡住无法移动
def do_death(c, car_agents, bike_agents, pedestrian_agents, destroy_list, statistics):
   
    update_time = time.time() #获取当前时间
    # 初始化新的代理列表，用于储存下一次更新后的代理和需要销毁的代理
    next_car_agents = []
    next_bike_agents = []
    next_pedestrian_agents = []
    new_destroy_list = []
    # 遍历车辆、自行车和行人代理列表
    # 对每一个代理检查 符合下面任何一个条件就删除
    # 1. 代理离开模拟区域
    # 2. 代理高度低于-10
    # 3. 代理类型是车辆或自行车，但不在SUMO网络占用区域内
    # 4. 代理路径点数量小于最小点数
    # 如果代理的速度小于stuck_speed，且代理被卡住的时间超过stuck_duration，则删除代理
    for (agents, next_agents) in zip([car_agents, bike_agents, pedestrian_agents], [next_car_agents, next_bike_agents, next_pedestrian_agents]):
        for agent in agents:
            delete = False
            if not delete and not c.bounds_occupancy.contains(get_position(agent.actor)):
                delete = True
            if not delete and get_position_3d(agent.actor).z < -10:
                delete = True
            if not delete and \
                    ((agent.type_tag in ['Car', 'Bicycle']) and not c.sumo_network_occupancy.contains(get_position(agent.actor))):
                delete = True
            if not delete and \
                    len(agent.path.route_points) < agent.path.min_points:
                delete = True
            if get_velocity(agent.actor).length() < c.args.stuck_speed:
                if agent.stuck_time is not None:
                    if update_time - agent.stuck_time >= c.args.stuck_duration:
                        if agents == car_agents:
                            statistics.stuck_num_cars += 1
                        elif agents == bike_agents:
                            statistics.stuck_num_bikes += 1
                        elif agents == pedestrian_agents:
                            statistics.stuck_num_pedestrians += 1
                        delete = True
                else:
                    agent.stuck_time = update_time
            else:
                agent.stuck_time = None
            # 如果代理需要删除，则将其id添加到销毁列表中
            if delete:
                new_destroy_list.append(agent.actor.id)
            else:
                next_agents.append(agent)
    # 更新统计数据
    return (next_car_agents, next_bike_agents, next_pedestrian_agents, destroy_list + new_destroy_list, statistics)

# 计算代理的平均速度，并更新统计信息
def do_speed_statistics(c, car_agents, bike_agents, pedestrian_agents, statistics):
    avg_speed_cars = 0.0
    avg_speed_bikes = 0.0
    avg_speed_pedestrians = 0.0

    for agent in car_agents:
        avg_speed_cars += get_velocity(agent.actor).length()

    for agent in bike_agents:
        avg_speed_bikes += get_velocity(agent.actor).length()

    for agent in pedestrian_agents:
        avg_speed_pedestrians += get_velocity(agent.actor).length()
    # 累积速度除以代理数量，得到平均速度
    if len(car_agents) > 0:
        avg_speed_cars /= len(car_agents)

    if len(bike_agents) > 0:
        avg_speed_bikes /= len(bike_agents)
        
    if len(pedestrian_agents) > 0:
        avg_speed_pedestrians /= len(pedestrian_agents)

    statistics.avg_speed_cars = avg_speed_cars
    statistics.avg_speed_bikes = avg_speed_bikes
    statistics.avg_speed_pedestrians = avg_speed_pedestrians
    # 返回更新后的统计数据
    return statistics

# 检测碰撞
def do_collision_statistics(c, timestamp, log_file):
    actors = c.world.get_actors() # 获取场景中的所有演员
    actors = [a for a in actors if is_car(a) or is_bike(a) or is_pedestrian(a)] # 筛选出车辆、自行车和行人
    bounding_boxes = [carla.OccupancyMap(get_bounding_box_corners(actor)) for actor in actors] # 获取所有演员的轴对齐边界框
    # 初始化碰撞列表，用于记录每个演员是否发生碰撞，初始值为0
    collisions = [0 for _ in range(len(actors))]
    # 遍历所有演员，检查是否发生碰撞，如果两者的边界框相交，则认为发生碰撞
    for i in range(len(actors) - 1):
        if collisions[i] == 1:
            continue
        for j in range(i + 1, len(actors)):
            if bounding_boxes[i].intersects(bounding_boxes[j]):
                collisions[i] = 1
                collisions[j] = 1
    # 将时间戳和碰撞列表写入日志文件，计算碰撞的比例
    log_file.write('{} {}\n'.format(timestamp, 0.0 if len(collisions) == 0 else float(sum(collisions)) / len(collisions)))
    log_file.flush()
    os.fsync(log_file)

# 在GAMMA中处理所有的代理（车辆、自行车和行人）
def do_gamma(c, car_agents, bike_agents, pedestrian_agents, destroy_list):
        # 将所有的代理（车辆、自行车和行人）合并到一个列表中
    agents = car_agents + bike_agents + pedestrian_agents
    # 创建一个查找表，用于快速访问代理对象
    agents_lookup = {}
    for agent in agents:
        agents_lookup[agent.actor.id] = agent
        agent.occupied_vertex = get_position(agent.actor)
        agent.selected_vertex = None
        agent.other_vehicles = [a for a in agents if a != agent and a.type_tag in ['Car', 'Bicycle']]
    # 初始化下一次迭代需要处理的代理列表和代理的GAMMA id列表
    next_agents = []
    next_agent_gamma_ids = []
    new_destroy_list = []# 初始化要销毁的代理列表
    # 如果存在代理，初始化GAMMA模拟器
    if len(agents) > 0:
        gamma = carla.RVOSimulator() # RVOSimulator是一个用于模拟多个代理之间的交互的模拟器
        
        gamma_id = 0

         # For external agents not tracked.处理在代理表中未跟踪的外部代理
        for actor in c.world.get_actors():
            if actor.id not in agents_lookup:
                if isinstance(actor, carla.Vehicle):
                    if is_bike(actor):
                        type_tag = 'Bicycle'
                    else:
                        type_tag = 'Car'
                    bounding_box_corners = get_vehicle_bounding_box_corners(actor)
                elif isinstance(actor, carla.Walker):
                    type_tag = 'People'
                    bounding_box_corners = get_pedestrian_bounding_box_corners(actor)
                else:
                    continue

                agent_params = carla.AgentParams.get_default(type_tag)
                if type_tag == 'Bicycle':                                                                                                                        
                    agent_params.max_speed = c.args.speed_bike
                elif type_tag == 'Car':
                    agent_params.max_speed = c.args.speed_car
                elif type_tag == 'People':
                    agent_params.max_speed = c.args.speed_pedestrian
                # 添加代理到GAMMA中，并设置代理的位置、速度、朝向、边界框和预期速度
                gamma.add_agent(agent_params, gamma_id) 
                gamma.set_agent_position(gamma_id, get_position(actor))
                gamma.set_agent_velocity(gamma_id, get_velocity(actor))
                gamma.set_agent_heading(gamma_id, get_forward_direction(actor))
                gamma.set_agent_bounding_box_corners(gamma_id, bounding_box_corners)
                gamma.set_agent_pref_velocity(gamma_id, get_velocity(actor))
                gamma_id += 1 # 更新GAMMA id，方便为下一个代理分配id

        # For tracked agents.处理被跟踪的代理
        for agent in agents:
            actor = agent.actor
            occupied_vertices = c.occupied_vertex
            selected_vertices = c.selected_vertex
            # Declare variables.初始变量
            is_valid = True # 代理是否有效
            pref_vel = None # 代理的预期速度
            path_forward = None # 代理的路径方向
            bounding_box_corners = None # 代理的边界框
            lane_constraints = None # 代理的车道约束
            # 如果代理是车辆或自行车
            # Update path, check validity, process variables.
            if agent.type_tag == 'Car' or agent.type_tag == 'Bicycle':
                position = get_position(actor)
                # Lane change if possible.生成随机数，如果小于预设车辆换道概率，则换道
                if c.rng.uniform(0.0, 1.0) <= c.args.lane_change_probability:
                    # 换道的过程是：获取代理的最近的路径点，然后获取下一个路径点的候选路径，如果候选路径不为空，则随机选择一个路径
                    new_path_candidates = c.sumo_network.get_next_route_paths(
                            c.sumo_network.get_nearest_route_point(position),
                            agent.path.min_points - 1, agent.path.interval)
                    if len(new_path_candidates) > 0:
                        new_path = SumoNetworkAgentPath(c.rng.choice(new_path_candidates)[0:agent.path.min_points], 
                                agent.path.min_points, agent.path.interval)
                        agent.path = new_path
                # Cut, resize, check.用resize调整代理的路径，使其适应SUMO网络
                if not agent.path.resize(c.sumo_network):
                    is_valid = False
                else:
                    agent.path.cut(c.sumo_network, position) # 如果调整成功会用cut来切割路径，删除最近路线点之前的所有路线点，从而使路径的起始点是代理的当前位置
                    if not agent.path.resize(c.sumo_network):
                        is_valid = False
                # Calculate variables.
                if is_valid:
                    # 获取代理路径上的目标位置，这个位置是路径上离代理当前位置第五个点的位置
                    # 计算当前位置到目标位置的向量，转换为单位向量，就是速度方向
                    target_position = agent.path.get_position(c.sumo_network, 5)  ## to check
                    velocity = (target_position - position).make_unit_vector()
                    # 计算预期速度，代理速度方向向量乘以预期速度
                    pref_vel = agent.preferred_speed * velocity
                    # 计算路径上的前向向量，路径上的第一个点到第二个点的向量，并转换为单位向量
                    path_forward = (agent.path.get_position(c.sumo_network, 1) - 
                            agent.path.get_position(c.sumo_network, 0)).make_unit_vector()
                    # 获取代理的边界框
                    bounding_box_corners = get_vehicle_bounding_box_corners(actor)
                    # 获取代理的车道约束
                    lane_constraints = get_lane_constraints(c.sidewalk, position, path_forward)
            elif agent.type_tag == 'People':
                position = get_position(actor)
                # Cut, resize, check.
                if not agent.path.resize(c.sidewalk, c.args.cross_probability):
                    is_valid = False
                else:
                    agent.path.cut(c.sidewalk, position)
                    if not agent.path.resize(c.sidewalk, c.args.cross_probability):
                        is_valid = False
                # Calculate pref_vel.
                if is_valid:
                    target_position = agent.path.get_position(c.sidewalk, 0)
                    velocity = (target_position - position).make_unit_vector()
                    pref_vel = agent.preferred_speed * velocity
                    path_forward = carla.Vector2D(0, 0) # Irrelevant for pedestrian.
                    bounding_box_corners = get_pedestrian_bounding_box_corners(actor)
            
            # Add info to GAMMA.
            if pref_vel:
                gamma.add_agent(carla.AgentParams.get_default(agent.type_tag), gamma_id)
                gamma.set_agent_position(gamma_id, get_position(actor))
                gamma.set_agent_velocity(gamma_id, get_velocity(actor))
                gamma.set_agent_heading(gamma_id, get_forward_direction(actor))
                gamma.set_agent_bounding_box_corners(gamma_id, bounding_box_corners)
                gamma.set_agent_pref_velocity(gamma_id, pref_vel)
                gamma.set_agent_path_forward(gamma_id, path_forward)
                if lane_constraints is not None:
                    # Flip LR -> RL since GAMMA uses right-handed instead.
                    gamma.set_agent_lane_constraints(gamma_id, lane_constraints[1], lane_constraints[0])  
                if agent.behavior_type is not -1:
                    gamma.set_agent_behavior_type(gamma_id, agent.behavior_type)
                next_agents.append(agent)
                next_agent_gamma_ids.append(gamma_id)
                gamma_id += 1
            else:
                new_destroy_list.append(agent.actor.id)

            if agent.behavior_type is -1:
                agent.control_velocity = get_ttc_vel(agent, agents, pref_vel)

        start = time.time()   # 记录开始时间     
        gamma.do_step() # 在GAMMA中进行一步模拟
        # 更新那些behavior_type不等于-1或者control_velocity为None的代理的控制速度。这个新的控制速度是从GAMMA模型中获取的。
        for (agent, gamma_id) in zip(next_agents, next_agent_gamma_ids):
            if agent.behavior_type is not -1 or agent.control_velocity is None:
                agent.control_velocity = gamma.get_agent_velocity(gamma_id)
    # 筛选出不同类型的代理
    next_car_agents = [a for a in next_agents if a.type_tag == 'Car']
    next_bike_agents = [a for a in next_agents if a.type_tag == 'Bicycle']
    next_pedestrian_agents = [a for a in next_agents if a.type_tag == 'People']
    # 合并销毁列表
    next_destroy_list = destroy_list + new_destroy_list
    # 更新控制速度到crowd_service中
    c.crowd_service.acquire_control_velocities()
    # 将next_agents列表中的每个代理的ID、类型标签、控制速度、预期速度和转向角范围作为一个元组添加到control_velocities列表中
    c.crowd_service.control_velocities = [
            (agent.actor.id, agent.type_tag, agent.control_velocity, agent.preferred_speed, agent.steer_angle_range)
            for agent in next_agents]
    # 释放控制速度列表
    c.crowd_service.release_control_velocities()
    # 更新本地意图到crowd_service中
    local_intentions = []
    # 如果代理的类型标签是People，则将代理的ID、类型标签、路径点和路径方向作为一个元组添加到local_intentions列表中
    for agent in next_agents:
        if agent.type_tag == 'People':
            local_intentions.append((agent.actor.id, agent.type_tag, agent.path.route_points[0], agent.path.route_orientations[0]))
        else:
            local_intentions.append((agent.actor.id, agent.type_tag, agent.path.route_points[0]))

    c.crowd_service.acquire_local_intentions()
    c.crowd_service.local_intentions = local_intentions
    c.crowd_service.release_local_intentions()

    return (next_car_agents, next_bike_agents, next_pedestrian_agents, next_destroy_list)

# 计算给定代理的期望速度
def get_ttc_vel(agent, agents, pref_vel):
    try:
        if agent:
            vel_to_exe = pref_vel
            if not vel_to_exe: # path is not ready.路径还没准备好
                return None

            speed_to_exe = agent.preferred_speed
            for other_agent in agents:
                # 如果other_agent存在且其ID与agent的ID不同
                if other_agent and agent.actor.id != other_agent.actor.id:
                    s_f = get_velocity(other_agent.actor).length() # 计算other_agent的速度
                    d_f = (get_position(other_agent.actor) - get_position(agent.actor)).length() # 计算agent与other_agent之间的距离
                    d_safe = 5.0 # 安全距离
                    a_max = 3.0 # 最大加速度
                    # 根据TTC原理：
                    # 如果s_f * s_f + 2 * a_max * (d_f - d_safe)大于0，则计算速度
                    # 在当前速度和方向不变的情况下，在最大加速度的情况下，可以保证不碰撞的速度
                    # 即 v^2 - v_0^2 = 2 * a_max * (d_f - d_safe)
                    s = max(0, s_f * s_f + 2 * a_max * (d_f - d_safe))**0.5
                    speed_to_exe = min(speed_to_exe, s)
            # 获取agent的当前速度与预期速度的角度差
            cur_vel = get_velocity(agent.actor)
            angle_diff = get_signed_angle_diff(vel_to_exe, cur_vel)
            # 如果角度差大于30或小于-30，则将速度设置为当前速度和预期速度的平均值
            if angle_diff > 30 or angle_diff < -30:
                vel_to_exe = 0.5 * (vel_to_exe + cur_vel)
            # 计算最终速度
            vel_to_exe = vel_to_exe.make_unit_vector() * speed_to_exe

            return vel_to_exe
    # 捕捉异常
    except Exception as e:
        print(e)

    return None

# 用PID控制在模拟环境中的各种实体
# 接收控制器对象‘c’，速度PID积分、速度PID最后的误差、转向PID积分、转向PID最后的误差和最后更新时间
def do_control(c, speed_pid_integrals, speed_pid_last_errors, steer_pid_integrals, steer_pid_last_errors, pid_last_update_time):
    start = time.time()

    c.crowd_service.acquire_control_velocities()
    control_velocities = c.crowd_service.control_velocities
    c.crowd_service.release_control_velocities()

    commands = []
    cur_time = time.time()
    if pid_last_update_time is None:
        dt = 0.0
    else:
        dt = cur_time - pid_last_update_time
    # 遍历控制速度列表
    for (actor_id, type_tag, control_velocity, preferred_speed, steer_angle_range) in control_velocities:

        actor = c.world.get_actor(actor_id)
        if actor is None:
            if actor_id in speed_pid_integrals:
                del speed_pid_integrals[actor_id]
            if actor_id in speed_pid_last_errors:
                del speed_pid_last_errors[actor_id]
            if actor_id in steer_pid_integrals:
                del steer_pid_integrals[actor_id]
            if actor_id in steer_pid_last_errors:
                del steer_pid_last_errors[actor_id]
            continue
        
        # 获取实体的当前速度与控制速度之间的角度差
        cur_vel = get_velocity(actor)
        
        angle_diff = get_signed_angle_diff(control_velocity, cur_vel)
        if angle_diff > 30 or angle_diff < -30:
            target_speed = 0.5 * (control_velocity + cur_vel)

        # 如果实体是汽车或自行车
        # 获取实体的速度、目标速度、速度PID参数、转向PID参数
        # 根据实体的类型和ID，从相应的PID配置文件中获取速度和转向的PID参数。
        # 然后，函数会根据实体的类型和当前速度，对目标速度进行裁剪，以防止在GAMMA突然改变时PID控制器的不稳定。
        if type_tag == 'Car' or type_tag == 'Bicycle':
            speed = get_velocity(actor).length()
            target_speed = control_velocity.length()
            control = actor.get_control()
            (speed_kp, speed_ki, speed_kd) = get_car_speed_pid_profile(actor.type_id) if type_tag == 'Car' else get_bike_speed_pid_profile(actor.type_id)
            (steer_kp, steer_ki, steer_kd) = get_car_steer_pid_profile(actor.type_id) if type_tag == 'Car' else get_bike_steer_pid_profile(actor.type_id)
            
            # Clip to stabilize PID against sudden changes in GAMMA.
            if type_tag == 'Car':
                target_speed = np.clip(target_speed, speed - 2.0, min(speed + 1.0, c.args.speed_car)) 
            if type_tag == 'Bicycle':
                target_speed = np.clip(target_speed, speed - 1.5, min(speed + 1.0, c.args.speed_bike)) 
            heading = math.atan2(cur_vel.y, cur_vel.x)
            heading_error = np.deg2rad(get_signed_angle_diff(control_velocity, cur_vel))
            target_heading = heading + heading_error
            
            # Calculate error.计算速度和朝向的误差
            speed_error = target_speed - speed
            heading_error = target_heading - heading
            # heading_error = np.clip(heading_error, -1.5, 1.5)

            # Add to integral. Clip to stablize integral term.将误差添加到积分中，以稳定积分项
            speed_pid_integrals[actor_id] += np.clip(speed_error, -0.3 / speed_kp, 0.3 / speed_kp) * dt 
            steer_pid_integrals[actor_id] += np.clip(heading_error, -0.3 / steer_kp, 0.3 / steer_kp) * dt 

            # Clip integral to prevent integral windup.对积分进行裁剪，以防止积分过载
            steer_pid_integrals[actor_id] = np.clip(steer_pid_integrals[actor_id], -0.02, 0.02)

            # Calculate output.计算出速度和转向控制量
            speed_control = speed_kp * speed_error + speed_ki * speed_pid_integrals[actor_id]
            steer_control = steer_kp * heading_error + steer_ki * steer_pid_integrals[actor_id]
            # 如果pid_last_update_time不是None，并且实体的ID在上一次的误差字典中，那么控制量会增加一个由当前误差和上一次误差之差除以时间差dt得到的项。
            if pid_last_update_time is not None and actor_id in speed_pid_last_errors:
                speed_control += speed_kd * (speed_error - speed_pid_last_errors[actor_id]) / dt
            if pid_last_update_time is not None and actor_id in steer_pid_last_errors:
                steer_control += steer_kd * (heading_error - steer_pid_last_errors[actor_id]) / dt

            # Update history.更新上一次的误差字典
            speed_pid_last_errors[actor_id] = speed_error
            steer_pid_last_errors[actor_id] = heading_error

            # Set control.设置实体的控制状态
            if speed_control >= 0:# 如果速度控制量大于等于0
                control.throttle = speed_control # 油门设置成速度控制量
                control.brake = 0.0 # 刹车设置成0
                control.hand_brake = False # 手刹设置成False
            else:
                control.throttle = 0.0 # 油门设置成0
                control.brake = -speed_control # 刹车设置成速度控制量的相反数
                control.hand_brake = False # 手刹设置成False
            # 转向被设置为转向控制量，但是会被裁剪到-1.0和1.0之间。然后，函数将实体的控制状态添加到命令列表中。
            control.steer = np.clip( steer_control, -1.0, 1.0)
            control.manual_gear_shift = True # DO NOT REMOVE: Reduces transmission lag.
            control.gear = 1 # DO NOT REMOVE: Reduces transmission lag.
            
            # Append to commands.将实体的控制状态添加到命令列表中
            commands.append(carla.command.ApplyVehicleControl(actor_id, control))
        # 如果实体是行人
        # 计算出实体的速度，并创建一个行人控制对象，然后将其添加到命令列表中。
        elif type_tag == 'People':
            velocity = np.clip(control_velocity.length(), 0.0, preferred_speed) * control_velocity.make_unit_vector()
            control = carla.WalkerControl(carla.Vector3D(velocity.x, velocity.y), 1.0, False)
            commands.append(carla.command.ApplyWalkerControl(actor_id, control))

    c.client.apply_batch(commands)# 客户端批量命令
 
    return cur_time # New pid_last_update_time.

# 在无限循环中，不断地生成和销毁代理
# 接收一个参数对象‘args’，其中包含了创建Context对象所需要的所有信息
def spawn_destroy_loop(args):
    try:
        # Wait for crowd service.
        # 等待人群服务的启动
        time.sleep(3)
        c = Context(args)

        # 将模拟环境的边界上传到crowd_service中
        c.crowd_service.simulation_bounds = (c.bounds_min, c.bounds_max)

        last_bounds_update = None # 上一次边界更新的时间
        
        print('Spawn-destroy loop running.')
        
        while True:
            start = time.time() # 记录当前时间

            # Download bounds
            # 如果这是循环的第一次迭代，或者自上次更新边界以来已经过去了1秒以上，那么就会从crowd_service中下载新的边界
            if last_bounds_update is None or start - last_bounds_update > 1.0:
                new_bounds = c.crowd_service.simulation_bounds
                # 如果新的边界与当前边界不同，则会更新边界
                if (new_bounds[0] is not None and new_bounds[0] != c.bounds_min) or \
                        (new_bounds[1] is not None and new_bounds[1] != c.bounds_max):
                    c.bounds_min = new_bounds[0]
                    c.bounds_max = new_bounds[1]
                    # 重新计算与新边界相交的SUMO网络占用区域和人行道占用区域
                    c.bounds_occupancy = carla.OccupancyMap(c.bounds_min, c.bounds_max)
                    c.sumo_network_spawn_segments = c.sumo_network_segments.intersection(carla.OccupancyMap(c.bounds_min, c.bounds_max))
                    c.sumo_network_spawn_segments.seed_rand(c.rng.getrandbits(32))
                    c.sidewalk_spawn_segments = c.sidewalk_segments.intersection(carla.OccupancyMap(c.bounds_min, c.bounds_max))
                    c.sidewalk_spawn_segments.seed_rand(c.rng.getrandbits(32))
                last_bounds_update = time.time()
            # 调用生成和销毁代理的函数
            do_spawn(c)
            do_destroy(c)
            # 休眠时间，使得生成和销毁代理的速率不超过SPAWN_DESTROY_MAX_RATE
            time.sleep(max(0, 1 / SPAWN_DESTROY_MAX_RATE - (time.time() - start)))
            # print('({}) Spawn-destroy rate: {} Hz'.format(os.getpid(), 1 / max(time.time() - start, 0.001)))
    # 捕捉异常
    # 如果与人群服务的连接意外关闭，函数不会立即停止，而是会继续尝试生成和销毁对象
    except Pyro4.errors.ConnectionClosedError:
        pass

# 控制循环，接收一个参数对象‘args’，其中包含了创建Context对象所需要的所有信息
def control_loop(args):
    try:
        # 等待3s，crowd_service启动
        time.sleep(3)
        c = Context(args)
        print('Control loop running.')
        # 初始化速度和转向的PID积分和上一次的误差       
        speed_pid_integrals = defaultdict(float)
        speed_pid_last_errors = defaultdict(float)
        pid_last_update_time = None # 初始化上一次更新PID控制器的时间

        steer_pid_integrals = defaultdict(float)
        steer_pid_last_errors = defaultdict(float)
        # 在每次循环中，首先记录当前时间为start，
        # 然后调用do_control函数进行控制，
        # 并更新pid_last_update_time。do_control函数可能会根据当前的状态和PID控制器的参数来调整车辆的速度和转向
        while True:
            start = time.time()
            pid_last_update_time = do_control(c, speed_pid_integrals, speed_pid_last_errors, 
                    steer_pid_integrals, steer_pid_last_errors, pid_last_update_time)
            time.sleep(max(0, 1 / CONTROL_MAX_RATE - (time.time() - start))) # 20 Hz
            # print('({}) Control rate: {} Hz'.format(os.getpid(), 1 / max(time.time() - start, 0.001)))

    # 捕获异常并继续执行。这可能是因为在网络通信中，连接可能会意外断开，这时候可以通过捕获异常来避免程序崩溃。
    except Pyro4.errors.ConnectionClosedError:
        pass

# GAMMA循环，接收一个参数对象‘args’，其中包含了创建Context对象所需要的所有信息
def gamma_loop(args):
    path_data_file = open('dijkstra_path.csv', 'w', newline='')
    path_data_writer = csv.writer(path_data_file)
    path_data_writer.writerow(['Agent ID', 'Route Points', 'Velocity', 'Acceleration'])
    max_vehicles = 1000
    try:
        # Wait for crowd service.
        time.sleep(3)
        c = Context(args)
        print('GAMMA loop running.')
        # 初始化车辆、自行车和行人代理列表
        car_agents = []
        bike_agents = []
        pedestrian_agents = []
        # 打开统计信息文件，模式为‘写’
        statistics_file = open('statistics.log', 'w')
        statistics = Statistics(statistics_file)
        statistics.start_time = time.time()

        # 最后一次更新模拟边界的时间
        last_bounds_update = None
        
        # 记录更新的频率
        rate_statistics_start = None
        rate_statistics_count = 0
        rate_statistics_done = False
        
        while True:
            destroy_list = []           # 无限循环，不断地处理代理
            start = time.time()   

            for agent in car_agents + bike_agents + pedestrian_agents:
                if agent.path_data:
                    for data in agent.path_data:
                        path_data_writer.writerow([agent.actor.id, data['route_points'], data['velocity'], data['acceleration']])
                    agent.path_data = []  # 清空路径数据
    
            # Download bounds.
            # Required for death process.
            if last_bounds_update is None or start - last_bounds_update > 1.0:
                
                # 更新模拟边界和禁止区域 
                new_bounds = c.crowd_service.simulation_bounds  
                if (new_bounds[0] is not None and new_bounds[0] != c.bounds_min) or \
                        (new_bounds[1] is not None and new_bounds[1] != c.bounds_max):
                    c.bounds_min = new_bounds[0]
                    c.bounds_max = new_bounds[1]
                    c.bounds_occupancy = carla.OccupancyMap(c.bounds_min, c.bounds_max)
                
                new_forbidden_bounds = c.crowd_service.forbidden_bounds
                if (new_forbidden_bounds[0] is not None and new_forbidden_bounds[0] != c.forbidden_bounds_min) or \
                        (new_forbidden_bounds[1] is not None and new_forbidden_bounds[1] != c.forbidden_bounds_max):
                    c.forbidden_bounds_min = new_forbidden_bounds[0]
                    c.forbidden_bounds_max = new_forbidden_bounds[1]
                    c.forbidden_bounds_occupancy = carla.OccupancyMap(c.forbidden_bounds_min, c.forbidden_bounds_max)

                last_bounds_update = time.time()
            total_vehicles = len(car_agents) + len(bike_agents) + len(pedestrian_agents)
            if total_vehicles >= max_vehicles:
                break
            # TODO: Maybe an functional-immutable interface wasn't the best idea...也许一个功能性的不可变接口不是最好的主意？？？？

            # Do this first if not new agents from pull_new_agents will affect avg. speed.
            # 计算平均速度，并更新统计信息
            (statistics) = \
                    do_speed_statistics(c, car_agents, bike_agents, pedestrian_agents, statistics)
            # 生成新的代理，并更新统计信息
            (car_agents, bike_agents, pedestrian_agents, statistics) = \
                    pull_new_agents(c, car_agents, bike_agents, pedestrian_agents, statistics)
            # 执行GAMMA算法，计算每个代理的下一个位置，并将需要销毁的代理添加到destory_list销毁列表中
            (car_agents, bike_agents, pedestrian_agents, destroy_list) = \
                    do_gamma(c, car_agents, bike_agents, pedestrian_agents, destroy_list)
            # 执行death函数，销毁代理，并更新统计信息
            (car_agents, bike_agents, pedestrian_agents, destroy_list, statistics) = \
                    do_death(c, car_agents, bike_agents, pedestrian_agents, destroy_list, statistics)

            #statistics.write()
            # 更新销毁列表
            c.crowd_service.acquire_destroy_list()
            c.crowd_service.extend_destroy_list(destroy_list)
            c.crowd_service.release_destroy_list()
            time.sleep(max(0, 1 / GAMMA_MAX_RATE - (time.time() - start))) # 40 Hz
            # print('({}) GAMMA rate: {} Hz'.format(os.getpid(), 1 / max(time.time() - start, 0.001)))

            # 计算和打印模拟的运行频率
            """if not rate_statistics_done:
                print(len(car_agents), len(bike_agents), len(pedestrian_agents), time.time() - statistics.start_time)
                if rate_statistics_start is None:
                    if time.time() - statistics.start_time > 180:
                        rate_statistics_start = time.time()
                else:
                    rate_statistics_count += 1
                    if time.time() - rate_statistics_start > 300:
                        print('Rate statistics = {:2f} Hz'.format(float(rate_statistics_count) / (time.time() - rate_statistics_start)))
                        rate_statistics_done = True"""
            
    except Pyro4.errors.ConnectionClosedError:
        pass

class VehicleRecorder:
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.records = []

    def add_record(self, position, velocity, acceleration, steer):
        self.records.append({
            'position': position,
            'velocity': velocity,
            'acceleration': acceleration,
            'steer': steer
        })

    def save_to_csv(self, file_path):
        with open(file_path, 'w', newline='') as csvfile:
            fieldnames = ['vehicle_id', 'position', 'velocity', 'acceleration', 'steer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                record['vehicle_id'] = self.vehicle_id
                writer.writerow(record)

# 收集和记录模拟中的碰撞统计信息，接收一个参数对象‘args’，其中包含了创建Context对象所需要的所有信息
def collision_statistics_loop(args):
    try:
        # 等待3s，crowd_service启动
        time.sleep(3)
        c = Context(args)
        print('Collision statistics loop running.')
        
        statistics_file = open('statistics_collision.log', 'w')
        sim_start_time = time.time()

        # 在无限循环中，不断地收集和记录模拟中的碰撞统计信息    
        # 为了控制碰撞统计的频率，函数会在每次循环结束时暂停一段时间。
        while True:
            start = time.time()
            do_collision_statistics(c, time.time() - sim_start_time, statistics_file)   
            # 停止时间：COLLISION_STATISTICS_MAX_RATE是碰撞统计的最大频率，
            # time.time() - start是do_collision_statistics函数的执行时间
            time.sleep(max(0, 1 / COLLISION_STATISTICS_MAX_RATE - (time.time() - start))) # 40 Hz 
            # print('({}) Collision statistics rate: {} Hz'.format(os.getpid(), 1 / max(time.time() - start, 0.001)))
    # 捕捉异常
    # 函数会直接忽略这个异常并继续执行。这通常意味着与人群服务的连接被关闭了。
    except Pyro4.errors.ConnectionClosedError:
        pass

# 用于启动模拟的各个子进程，并启动一个Pyro4守护进程来提供crowd_service服务
def main(args):
    # 三个循环的子进程：spawn_destroy_loop、control_loop、gamma_loop、collision_statistics_loop（在原代码中并未启用）
    # 每一个都是守护进程，这意味着它们会在主进程结束时自动结束
    # start来启动这些子进程
    spawn_destroy_process = Process(target=spawn_destroy_loop, args=(args,))
    spawn_destroy_process.daemon = True
    spawn_destroy_process.start()

    control_process = Process(target=control_loop, args=(args,))
    control_process.daemon = True
    control_process.start()
    
    gamma_process = Process(target=gamma_loop, args=(args,))
    gamma_process.daemon = True
    gamma_process.start()


    collision_statistics_process = Process(target=collision_statistics_loop, args=(args,))
    collision_statistics_process.daemon = True
    collision_statistics_process.start()
    
    # 创建一个Pyro4守护进程，并注册了一个名为crowdservice.warehouse的人群服务
    # 它会在args.pyroport端口上监听，但不会在Pyro4的名字服务器上注册
    # """这个守护进程会监听指定的端口，等待其他进程通过Pyro4库来调用这个服务。注册服务后，守护进程会进入一个无限循环，等待和处理来自其他进程的请求
    Pyro4.Daemon.serveSimple(                                           
            {                                         
                CrowdService: "crowdservice.warehouse"                 
            },
            port=args.pyroport,
            ns=False)
    spawn_destroy_process.join()
    control_process.join()
    gamma_process.join()
    collision_statistics_process.join()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-pyp', '--pyroport',
        default=8100,
        type=int,
        help='TCP port for pyro4 to listen to (default: 8100)')
    argparser.add_argument(
        '-d', '--dataset',
        default='meskel_square',
        help='Name of dataset (default: meskel_square)')
    argparser.add_argument(
        '-s', '--seed',
        default='-1',
        help='Value of random seed (default: -1)',
        type=int)
    argparser.add_argument(
        '--collision',
        help='Enables collision for controlled agents',
        action='store_true')
    argparser.add_argument(
        '--num-car',
        default='60',
        help='Number of cars to spawn (default: 20)',
        type=int)
    argparser.add_argument(
        '--num-bike',
        default='0',
        help='Number of bikes to spawn (default: 20)',
        type=int)
    argparser.add_argument(
        '--num-pedestrian',
        default='0',
        help='Number of pedestrians to spawn (default: 20)',
        type=int)
    argparser.add_argument(
        '--speed-car',
        default='15',
        help='Mean preferred_speed of cars',
        type=float)
    argparser.add_argument(
        '--speed-bike',
        default='2.0',
        help='Mean preferred_speed of bikes',
        type=float)
    argparser.add_argument(
        '--speed-pedestrian',
        default='1.0',
        help='Mean preferred_speed of pedestrains',
        type=float)
    argparser.add_argument(
        '--clearance-car',
        default='10.0',
        help='Minimum clearance (m) when spawning a car (default: 10.0)',
        type=float)
    argparser.add_argument(
        '--clearance-bike',
        default='10.0',
        help='Minimum clearance (m) when spawning a bike (default: 10.0)',
        type=float)
    argparser.add_argument(
        '--clearance-pedestrian',
        default='1.0',
        help='Minimum clearance (m) when spawning a pedestrian (default: 1.0)',
        type=float)
    argparser.add_argument(
        '--lane-change-probability',
        default='0.0',
        help='Probability of lane change for cars and bikes (default: 0.0)',
        type=float)
    argparser.add_argument(
        '--cross-probability',
        default='0.1',
        help='Probability of crossing road for pedestrians (default: 0.1)',
        type=float)
    argparser.add_argument(
        '--stuck-speed',
        default='0.2',
        help='Maximum speed (m/s) for an agent to be considered stuck (default: 0.2)',
        type=float)
    argparser.add_argument(
        '--stuck-duration',
        default='5.0',
        help='Minimum duration (s) for an agent to be considered stuck (default: 5)',
        type=float)
    args = argparser.parse_args()
    main(args)
    

import copy
import sys
import os
import numpy as np
import pdb

class STRIPS:
    def __init__(self):
        self.state = np.array([1, 0,0, 1,0,1, 0,1,0, 0,0,0])
        self.goal = np.array([None, None,None, None,0,None, None,None,None, 0,1,0])
        self.num_objects = 2
        self.num_docks = 3
        self.operators = ['dock_objA_dockA', 'dock_objA_dockB', 'dock_objA_dockC', \
                    'dock_objB_dockA', 'dock_objB_dockB', 'dock_objB_dockC', \
                    'undock_objA_dockA', 'undock_objA_dockB', 'undock_objA_dockC', \
                    'undock_objB_dockA', 'undock_objB_dockB', 'undock_objB_dockC', \
                    'grasp_objA', 'grasp_objB']

    # predicates
    def gripper_free(self):
        # Is gripper currently free
        return True if self.state[0] else False

    def grasped(self, obj):
        # Is obj currently being grasped by Astrobee gripper
        # objA = 1, objB = 2
        return True if self.state[obj] else False

    def dock_clear(self, dock):
        # Is dock currently clear of any objects
        # dockA=1, dockB=2, dockC=3 
        return True if self.state[2+dock] else False

    def docked(self, obj, dock):
        # Is obj currently docked to dock
        if self.state[self.num_objects + self.num_docks + self.num_docks*(obj-1) + dock]:
            return True
        return False

    # operators
    def grasp(self, obj):
        if not self.gripper_free():
            return
        elif any([self.grasped(obj) for obj in range(1,self.num_objects+1)]):
            return
        self.state[0] = False     # set gripper_free False
        self.state[obj] = True    # grasp object
        return

    def dock(self, obj, dock):
        if not self.grasped(obj) or not self.dock_clear(dock): 
            return
        self.state[0] = True            # gripper_free now true 
        self.state[obj] = False         # ungrasp object
        self.state[2+dock] = False      # dock is no longer clear
        self.state[5 + 3*(obj-1) + dock] = True   # docked(obj,dock) now true

    def undock(self, obj, dock):
        if not self.gripper_free() or not self.docked(obj,dock):
            return
        self.state[0] = False     # gripper_free now false
        self.state[obj] = True    # obj now grasped
        self.state[2+dock] = True   # dock now clear 
        self.state[5 + 3*(obj-1) + dock] = False    # docked(obj,dock) now false

    def solved(self):
        for ii in range(self.goal.size):
            if self.goal[ii] is not None and self.state[ii] != self.goal[ii]:
                return False
        return True

class Node:
    def __init__(self, strip, parent=None, operator=None):
        self.strip = copy.copy(strip)
        self.parent = parent 
        self.operator = operator 
        self.depth = 0
        if parent:
            self.depth = parent.depth+1
        self.solved = strip.solved() 

    def act(self, operator): 
        strip = copy.deepcopy(self.strip)
        if operator == 'dock_objA_dockA':
            strip.dock(1,1)
        elif operator == 'dock_objA_dockB':
            strip.dock(1,2)
        elif operator == 'dock_objA_dockC':
            strip.dock(1,3)
        elif operator == 'dock_objB_dockA':
            strip.dock(2,1)
        elif operator == 'dock_objB_dockB':
            strip.dock(2,2)
        elif operator == 'dock_objB_dockC':
            strip.dock(2,3)
        elif operator == 'undock_objA_dockA':
            strip.undock(1,1)
        elif operator == 'undock_objA_dockB':
            strip.undock(1,2)
        elif operator == 'undock_objA_dockC':
            strip.undock(1,3)
        elif operator == 'undock_objB_dockA':
            strip.undock(2,1)
        elif operator == 'undock_objB_dockB':
            strip.undock(2,2)
        elif operator == 'undock_objB_dockC':
            strip.undock(2,3)
        elif operator == 'grasp_objA':
            strip.grasp(1)
        elif operator == 'grasp_objB':
            strip.grasp(2)
        new_node = Node(strip, parent=self, operator=operator)
        return new_node

def get_plan(leaf_node):
    plan = []
    if not leaf_node.solved:
        return plan 
    while leaf_node.parent:
        plan.append(leaf_node.operator)
        leaf_node = leaf_node.parent
    plan.reverse()
    return plan

class Queue:
    def __init__(self):
        self.list = []

    def queue(self, data):
        self.list.insert(0, data)

    def dequeue(self):
        if len(self.list) > 0:
            return self.list.pop()
        return None

    def is_empty(self):
        return True if len(self.list) == 0 else False

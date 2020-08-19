import copy
import sys
import os
import numpy as np
import pdb

class STRIPS:
    def __init__(self):
        self.state = np.array([1,0,0])
        self.goal = np.array([1,1,1])

    def cap_on(self):
        return True if self.state[0] else False

    def battery_in(self, battery_idx):
        return True if self.state[battery_idx] else False

    def place_cap(self):
        if self.cap_on():
            return
        self.state[0] = 1

    def remove_cap(self):
        if not self.cap_on():
            return
        self.state[0] = 0

    def insert(self, battery_idx):
        if self.battery_in(battery_idx) or self.cap_on():
            return
        self.state[battery_idx] = 1

    def solved(self):
        return all(self.state == self.goal)

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
        if operator == 'place_cap':
            strip.place_cap()
        elif operator == 'remove_cap':
            strip.remove_cap()
        elif operator == 'insert_0':
            strip.insert(1)
        elif operator == 'insert_1':
            strip.insert(2)
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

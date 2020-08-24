import copy
import sys
import os
import numpy as np
import pdb

sys.path.append('/home/acauligi/Software')
from casadi import *

from astrobee_strips import STRIPS, Node, Queue, get_plan
from scp import update_f, update_A, update_B, slerp
from double_integrator import compute_Ak, compute_Bk

class AstrobeeTAMP:
    def __init__(self, Xi, Xref, mode='double_integrator'):
        self.operators = ['dock_objA_dockA', 'dock_objA_dockB', 'dock_objA_dockC', \
                    'dock_objB_dockA', 'dock_objB_dockB', 'dock_objB_dockC', \
                    'undock_objA_dockA', 'undock_objA_dockB', 'undock_objA_dockC', \
                    'undock_objB_dockA', 'undock_objB_dockB', 'undock_objB_dockC', \
                    'grasp_objA', 'grasp_objB']

        self.N = 10 
        self.mode = mode
        self.Xi = Xi
        self.Xref = Xref

        if mode == 'double_integrator':
            self.n, self.m = 4, 2
        else:
            self.n, self.m = 13, 6

        self.dh = 0.05
        self.Xprev, self.Uprev = None, None

        self.R = np.eye(self.m)

        # environment parameters
        self.dock_locations = {}
        self.dock_locations['A'] = np.array([10.,20.])
        self.dock_locations['B'] = np.array([10.,30.])
        self.dock_locations['C'] = np.array([10.,40.])

        # robot parameters
        J = np.array([[0.1083, 0.0, 0.0],
                    [0.0, 0.1083, 0.0],
                    [0.0, 0.0, 0.1083]])
        Jxx, Jyy, Jzz = np.diag(J)
        Jinv = np.linalg.inv(J)
        mass = 7.0

        hard_limit_vel   = 5000. # 0.50
        hard_limit_accel = 1000. # 0.10
        hard_limit_omega = 45*np.pi/180
        hard_limit_alpha = 50*np.pi/180

        self.params = {}
        self.params['mass'] = mass
        self.params['J'] = J
        self.params['hard_limit_vel'] = hard_limit_vel
        self.params['hard_limit_accel'] = hard_limit_accel
        self.params['hard_limit_omega'] = hard_limit_omega
        self.params['hard_limit_alpha'] = hard_limit_alpha
        self.params['dh'] = self.dh

        # state box constraints
        if self.mode == 'double_integrator':
            self.Xlb = np.array([-np.inf,-np.inf,
                -hard_limit_vel/np.sqrt(2),-hard_limit_vel/np.sqrt(2)])
            self.Xub = np.array([np.inf,np.inf,
                hard_limit_vel/np.sqrt(2),hard_limit_vel/np.sqrt(2)])

            # control box constraints
            Jmin = np.min(np.diag(J))
            self.Ulb = np.array([-mass*hard_limit_accel/np.sqrt(2), -mass*hard_limit_accel/np.sqrt(2)])
            self.Uub = np.array([mass*hard_limit_accel/np.sqrt(2), mass*hard_limit_accel/np.sqrt(2)])
        else:
            self.Xlb = np.array([-np.inf,-np.inf,-np.inf,
                -1.0,-1.0,-1.0,-1.0,
                -hard_limit_vel/np.sqrt(3),-hard_limit_vel/np.sqrt(3),-hard_limit_vel/np.sqrt(3),
                -hard_limit_omega/np.sqrt(3),-hard_limit_omega/np.sqrt(3),-hard_limit_omega/np.sqrt(3)])
            self.Xub = np.array([np.inf,np.inf,np.inf,
                1.0,1.0,1.0,1.0,
                hard_limit_vel/np.sqrt(3),hard_limit_vel/np.sqrt(3),hard_limit_vel/np.sqrt(3),
                hard_limit_omega/np.sqrt(3),hard_limit_omega/np.sqrt(3),hard_limit_omega/np.sqrt(3)])

            # control box constraints
            Jmin = np.min(np.diag(J))
            self.Ulb = np.array([-mass*hard_limit_accel/np.sqrt(3), -mass*hard_limit_accel/np.sqrt(3), -mass*hard_limit_accel/np.sqrt(3), -Jmin*hard_limit_alpha/np.sqrt(3), -Jmin*hard_limit_alpha/np.sqrt(3), -Jmin*hard_limit_alpha/np.sqrt(3)])
            self.Uub = np.array([mass*hard_limit_accel/np.sqrt(3), mass*hard_limit_accel/np.sqrt(3), mass*hard_limit_accel/np.sqrt(3),  Jmin*hard_limit_alpha/np.sqrt(3), Jmin*hard_limit_alpha/np.sqrt(3), Jmin*hard_limit_alpha/np.sqrt(3)])

        self.cost = 0.
        self.w, self.lbw, self.ubw = None, None, None 
        self.g, self.lbg, self.ubg = None, None, None 

    def init_straightline(self):
        N_plan = len(self.X)
        self.Xprev, self.Uprev = np.zeros((self.n, N_plan)), np.zeros((self.m, N_plan-1))

        if self.mode == 'double_integrator':
            for ii in range(self.n):
                self.Xprev[ii,:] = np.linspace(self.Xi[ii], np.array(self.Xref)[ii], num=N_plan).flatten()
        else:
            for ii in range(3):
                self.Xprev[ii,:] = np.linspace(self.Xi[ii], np.array(self.Xref)[ii], num=N_plan).flatten()
                self.Xprev[7+ii,:] = np.linspace(self.Xi[7+ii], np.array(self.Xref)[7+ii], num=N_plan).flatten()
                self.Xprev[10+ii,:] = np.linspace(self.Xi[10+ii], np.array(self.Xref)[10+ii], num=N_plan).flatten()

            qi = self.Xi[3:7].flatten()
            qf = np.array(self.Xref)[3:7].flatten()
            for idx, time in enumerate(np.linspace(0.,1., N_plan)):
              self.Xprev[3:7, idx] = slerp(qi, qf, time)
        return

    def update_dynamics(self):
        plan_len = len(self.plan)
        N_plan = self.N*plan_len

        if self.mode == 'double_integrator':
            self.Aks, self.Bks = [], []
            for ii in range(N_plan-1):
                self.Aks.append(compute_Ak(self.Xprev[:,ii], self.Uprev[:,ii], self.params))
                self.Bks.append(compute_Bk(self.Xprev[:,ii], self.Uprev[:,ii], self.params))
        else:
            self.fs, self.As, self.Bs = [], [], []
            for ii in range(N_plan-1):
                self.fs.append(update_f(self.Xprev[:,ii], self.Uprev[:,ii], self.params))
                self.As.append(update_A(self.Xprev[:,ii], self.Uprev[:,ii], self.params))
                self.Bs.append(update_B(self.Xprev[:,ii], self.Uprev[:,ii], self.params))

    def unroll_plan(self): 
        self.cost = 0.
        self.X, self.U = [], []
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []

        for operator in self.plan:
            w_loc, lbw_loc, ubw_loc, g_loc, lbg_loc, ubg_loc, X_loc, U_loc, J_loc = self.operator_cons(operator)
            if len(w_loc) == 0:
                continue
            self.cost += (J_loc + self.lqr_cost(X_loc, U_loc, self.Xref))
            self.w += [*w_loc]
            self.lbw += [*lbw_loc]
            self.ubw += [*ubw_loc]
            self.g += [*g_loc]
            self.lbg += [*lbg_loc]
            self.ubg += [*ubg_loc]
            self.X.extend(X_loc)
            self.U.extend(U_loc)

        # Add dynamics constraints
        self.init_straightline()
        self.update_dynamics()
        self.add_dyn_con()

        w_dyn, lbw_dyn, ubw_dyn, g_dyn, lbg_dyn, ubg_dyn = self.add_dyn_con()
        self.w += [*w_dyn]
        self.lbw += [*lbw_dyn]
        self.ubw += [*ubw_dyn]
        self.g += [*g_dyn]
        self.lbg += [*lbg_dyn]
        self.ubg += [*ubg_dyn]

        # Add initial condition constraint
        self.lbw[:self.n] = self.Xi.tolist()
        self.ubw[:self.n] = self.Xi.tolist()

    def construct_problem(self):
        prob = {'f':self.cost, 'x':vertcat(*self.w), 'g': vertcat(*self.g)}
        self.solver = nlpsol('S', 'ipopt', prob)
        return

    def solve_problem(self, use_guess=False, w0=None):
        self.soln = None
        if use_guess:
            # TODO(acauligi)
            pass
            self.soln = self.solver(x0=w0, lbx=self.lbw, ubx=self.ubw, lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))
        else:
            self.soln = self.solver(lbx=self.lbw, ubx=self.ubw, lbg=vertcat(*self.lbg), ubg=vertcat(*self.ubg))

        return True if self.solver.stats()['success'] else False

    def operator_cons(self, operator):
        if operator == 'dock_objA_dockA':
            return self.dock_objA_dockA()
        elif operator == 'dock_objA_dockB':
            return self.dock_objA_dockB()
        elif operator == 'dock_objA_dockC':
            return self.dock_objA_dockC()
        elif operator == 'dock_objB_dockA':
            return self.dock_objB_dockA()
        elif operator == 'dock_objB_dockB':
            return self.dock_objB_dockB()
        elif operator == 'dock_objB_dockC':
            return self.dock_objB_dockC()
        elif operator == 'undock_objA_dockA':
            return self.undock_objA_dockA()
        elif operator == 'undock_objA_dockB':
            return self.undock_objA_dockB()
        elif operator == 'undock_objA_dockC':
            return self.undock_objA_dockC()
        elif operator == 'undock_objB_dockA':
            return self.undock_objB_dockA()
        elif operator == 'undock_objB_dockB':
            return self.undock_objB_dockB()
        elif operator == 'undock_objB_dockC':
            return self.undock_objB_dockC()
        elif operator == 'grasp_objA':
            return self.grasp_objA()
        elif operator == 'grasp_objB':
            return self.grasp_objB()

    def create_vars(self, N, lb, ub, tag='X_'):
        n = lb.size 
        w, lbw, ubw = [], [], []
        X = []
        for ii in range(N):
              Xn = MX.sym(tag+str(ii), n)
              X += [Xn]
              w += [Xn]
              lbw += lb.tolist()
              ubw += ub.tolist()
        return w, lbw, ubw

    def lqr_cost(self, X, U, Xref):
        N = len(X)
        J = 0.
        for ii in range(N):
            stage_cost = casadi.dot(X[ii] - Xref, X[ii] - Xref)
            J += stage_cost
        for ii in range(N-1):
            J += casadi.dot(U[ii], self.R @ U[ii])
        return J

    def generate_segment_vars(self, tag='X_'):
        X, U = [], []
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X, lbx, ubx = self.create_vars(self.N, self.Xlb, self.Xub, tag='X_')
        w += [*X]
        lbw += [*lbx]
        ubw += [*ubx]

        U, lbu, ubu = self.create_vars(self.N, self.Ulb, self.Uub, tag='U_')
        w += [*U]
        lbw += [*lbu]
        ubw += [*ubu]

        return w, lbw, ubw, g, lbg, ubg, X, U

    def add_dyn_con(self):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        plan_len = len(self.plan)
        N_plan = self.N*plan_len

        X0 = self.w[0]
        for ii in range(N_plan-1):
            if self.mode == 'double_integrator':
                Ak, Bk = self.Aks[ii], self.Bks[ii]
                Q = Ak @ self.X[ii] + Bk @ self.U[ii] - self.X[ii+1]
            else:
                Ak, Bk = np.eye(self.n) + self.dh*self.As[ii], self.dh*self.Bs[ii]
                ck = self.dh * (self.fs[ii] - self.As[ii] @ self.Xprev[:,ii] - self.Bs[ii] @ self.Uprev[:,ii])

                Q = Ak @ self.X[ii] + Bk @ self.U[ii] + ck - self.X[ii+1]
            g += [Q]
            lbg += [np.zeros(self.n).tolist()]
            ubg += [np.zeros(self.n).tolist()]

        return w, lbw, ubw, g, lbg, ubg

    def add_grasp_con(self, obj_id):
        # TODO(acauligi): compute forward kinematics given
        # reference trajectory and add equality constraint
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        for ii in range(N):
            g += []
            lbg += []
            ubg += []
        return w, lbw, ubw, g, lbg, ubg, 0.

    def add_dock_con(self, obj_id, dock_coord):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        # TODO(acauligi): add constraint to end of trajectory

        return w, lbw, ubw, g, lbg, ubg

    def dock_objA_dockA(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def dock_objA_dockB(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def dock_objA_dockC(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def dock_objB_dockA(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def dock_objB_dockB(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def dock_objB_dockC(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def undock_objA_dockA(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def undock_objA_dockB(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def undock_objA_dockC(self):
        w, lbw, ubw, g, lbg, ubg, J = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def undock_objB_dockA(self):
        w, lbw, ubw, g, lbg, ubg, J = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def undock_objB_dockB(self):
        w, lbw, ubw, g, lbg, ubg, J = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def undock_objB_dockC(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def grasp_objA(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

    def grasp_objB(self):
        w, lbw, ubw, g, lbg, ubg, X, U = self.generate_segment_vars()
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X, U, J

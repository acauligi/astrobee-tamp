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
        self.dock_loc = {}
        self.dock_loc['A'] = np.array([10., 15.])
        self.dock_loc['B'] = np.array([20., 15.])
        self.dock_loc['C'] = np.array([30., 15.])

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

        self.arm_length = 0.1

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
        self.Xg = []
        self.w, self.lbw, self.ubw = [], [], []
        self.g, self.lbg, self.ubg = [], [], []

        # Create decision variables for free-flyer
        # for operator in self.plan:
            # w_loc, lbw_loc, ubw_loc, g_loc, lbg_loc, ubg_loc, X_loc, U_loc \
            #     = self.generate_segment_vars()
            # self.w += [*w_loc]
            # self.lbw += [*lbw_loc]
            # self.ubw += [*ubw_loc]
            # self.g += [*g_loc]
            # self.lbg += [*lbg_loc]
            # self.ubg += [*ubg_loc]

        for operator in self.plan:
            X, lbx, ubx = self.create_vars(self.N, self.Xlb, self.Xub, tag='X_')
            self.w += [*X]
            self.lbw += [*lbx]
            self.ubw += [*ubx]
            self.X.extend(X)

        for operator in self.plan:
            U, lbu, ubu = self.create_vars(self.N, self.Ulb, self.Uub, tag='U_')
            self.w += [*U]
            self.lbw += [*lbu]
            self.ubw += [*ubu]
            self.U.extend(U)

        # Create additional vars and constraints for grasping
        for ii, operator in enumerate(self.plan):
            X_loc, U_loc = self.X[self.N*ii:self.N*(ii+1)], self.U[self.N*ii:self.N*(ii+1)] 

            w_loc, lbw_loc, ubw_loc, \
              g_loc, lbg_loc, ubg_loc, \
              Xg_loc, J_loc = self.operator_cons(operator, X_loc, U_loc)
            if len(w_loc) == 0:
                continue
            self.cost += (J_loc + self.lqr_cost(X_loc, U_loc, self.Xref))
            self.w += [*w_loc]
            self.lbw += [*lbw_loc]
            self.ubw += [*ubw_loc]
            self.g += [*g_loc]
            self.lbg += [*lbg_loc]
            self.ubg += [*ubg_loc]

            if Xg_loc is not None:
                self.Xg.extend(Xg_loc)

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

    def operator_cons(self, operator, X_loc, U_loc):
        if operator == 'dock_objA_dockA':
            return self.dock_objA_dockA(X_loc, U_loc)
        elif operator == 'dock_objA_dockB':
            return self.dock_objA_dockB(X_loc, U_loc)
        elif operator == 'dock_objA_dockC':
            return self.dock_objA_dockC(X_loc, U_loc)
        elif operator == 'dock_objB_dockA':
            return self.dock_objB_dockA(X_loc, U_loc)
        elif operator == 'dock_objB_dockB':
            return self.dock_objB_dockB(X_loc, U_loc)
        elif operator == 'dock_objB_dockC':
            return self.dock_objB_dockC(X_loc, U_loc)
        elif operator == 'undock_objA_dockA':
            return self.undock_objA_dockA(X_loc, U_loc)
        elif operator == 'undock_objA_dockB':
            return self.undock_objA_dockB(X_loc, U_loc)
        elif operator == 'undock_objA_dockC':
            return self.undock_objA_dockC(X_loc, U_loc)
        elif operator == 'undock_objB_dockA':
            return self.undock_objB_dockA(X_loc, U_loc)
        elif operator == 'undock_objB_dockB':
            return self.undock_objB_dockB(X_loc, U_loc)
        elif operator == 'undock_objB_dockC':
            return self.undock_objB_dockC(X_loc, U_loc)
        elif operator == 'grasp_objA':
            return self.grasp_objA(X_loc, U_loc)
        elif operator == 'grasp_objB':
            return self.grasp_objB(X_loc, U_loc)

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

    def add_grasp_con(self, X, obj_X):
        # TODO(acauligi): compute forward kinematics given
        # reference trajectory and add equality constraint
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        for ii in range(len(X)):
            g += [X[ii][0] - obj_X[ii][0]]
            lbg += [0.]
            ubg += [0.]

            g += [X[ii][1]+self.arm_length - obj_X[ii][1]]
            lbg += [0.]
            ubg += [0.]
        return w, lbw, ubw, g, lbg, ubg, 0.

    def add_dock_con(self, obj_X, dock_loc):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        for ii in range(2):
            g += [obj_X[-1][ii] - dock_loc[ii]]
            lbg += [0.]
            ubg += [0.]

        return w, lbw, ubw, g, lbg, ubg

    def add_grasp_dock_con(self, X, dock_id):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        obj_X = None
        tag = 'obj{}_'.format(dock_id)

        obj_X, lb_obj, ub_obj = self.create_vars(self.N, self.Xlb, self.Xub, tag=tag)
        w.extend(obj_X)
        lbw.extend(lb_obj)
        ubw.extend(ub_obj)

        w_grasp, lbw_grasp, ubw_grasp, g_grasp, lbg_grasp, ubg_grasp, _  = self.add_grasp_con(X, obj_X)
        g.extend(g_grasp)
        lbg.extend(lbg_grasp)
        ubg.extend(ubg_grasp)

        w_dock, lbw_dock, ubw_dock, g_dock, lbg_dock, ubg_dock = self.add_dock_con(obj_X, self.dock_loc['A'])
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)
        return obj_X, w, lbw, ubw, g, lbg, ubg

    def dock_objA_dockA(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []
        J = 0.

        dock_id = 'A'
        X_objA, w_objA, lbw_objA, ubw_objA, g_objA, lbg_objA, ubg_objA= self.add_grasp_dock_con(X, dock_id)
        w.extend(w_objA)
        lbw.extend(lbw_objA)
        ubw.extend(ubw_objA)
        g.extend(g_objA)
        lbg.extend(lbg_objA)
        ubg.extend(ubg_objA)

        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def dock_objA_dockB(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objA = None
        J = 0.

        dock_id = 'B'
        X_objA, w_objA, lbw_objA, ubw_objA, g_objA, lbg_objA, ubg_objA= self.add_grasp_dock_con(X, dock_id)
        w.extend(w_objA)
        lbw.extend(lbw_objA)
        ubw.extend(ubw_objA)
        g.extend(g_objA)
        lbg.extend(lbg_objA)
        ubg.extend(ubg_objA)

        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def dock_objA_dockC(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objA = None
        J = 0.

        dock_id = 'C'
        X_objA, w_objA, lbw_objA, ubw_objA, g_objA, lbg_objA, ubg_objA= self.add_grasp_dock_con(X, dock_id)
        w.extend(w_objA)
        lbw.extend(lbw_objA)
        ubw.extend(ubw_objA)
        g.extend(g_objA)
        lbg.extend(lbg_objA)
        ubg.extend(ubg_objA)

        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def dock_objB_dockA(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.

        dock_id = 'A'
        X_objB, w_objB, lbw_objB, ubw_objB, g_objB, lbg_objB, ubg_objB= self.add_grasp_dock_con(X, dock_id)
        w.extend(w_objB)
        lbw.extend(lbw_objB)
        ubw.extend(ubw_objB)
        g.extend(g_objB)
        lbg.extend(lbg_objB)
        ubg.extend(ubg_objB)

        return w, lbw, ubw, g, lbg, ubg, X_objB, J

    def dock_objB_dockB(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.

        dock_id = 'B'
        X_objB, w_objB, lbw_objB, ubw_objB, g_objB, lbg_objB, ubg_objB= self.add_grasp_dock_con(X, dock_id)
        w.extend(w_objB)
        lbw.extend(lbw_objB)
        ubw.extend(ubw_objB)
        g.extend(g_objB)
        lbg.extend(lbg_objB)
        ubg.extend(ubg_objB)

        return w, lbw, ubw, g, lbg, ubg, X_objB, J

    def dock_objB_dockC(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.

        dock_id = 'C'
        X_objB, w_objB, lbw_objB, ubw_objB, g_objB, lbg_objB, ubg_objB= self.add_grasp_dock_con(X, dock_id)
        w.extend(w_objB)
        lbw.extend(lbw_objB)
        ubw.extend(ubw_objB)
        g.extend(g_objB)
        lbg.extend(lbg_objB)
        ubg.extend(ubg_objB)

        return w, lbw, ubw, g, lbg, ubg, X_objB, J

    def undock_objA_dockA(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objA = None
        J = 0.

        dock_id = 'A'
        obj_loc = self.dock_loc[dock_id]
        obj_loc[1] -= self.arm_length
        _, _, _, g_dock, lbg_dock, ubg_dock = self.add_dock_con(X, obj_loc)
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)

        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def undock_objA_dockB(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objA = None
        J = 0.

        dock_id = 'B'
        obj_loc = self.dock_loc[dock_id]
        obj_loc[1] -= self.arm_length
        _, _, _, g_dock, lbg_dock, ubg_dock = self.add_dock_con(X, obj_loc)
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)

        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def undock_objA_dockC(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objA = None
        J = 0.

        dock_id = 'C'
        obj_loc = self.dock_loc[dock_id]
        obj_loc[1] -= self.arm_length
        _, _, _, g_dock, lbg_dock, ubg_dock = self.add_dock_con(X, obj_loc)
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)

        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def undock_objB_dockA(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.

        dock_id = 'A'
        obj_loc = self.dock_loc[dock_id]
        obj_loc[1] -= self.arm_length
        _, _, _, g_dock, lbg_dock, ubg_dock = self.add_dock_con(X, obj_loc)
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)

        return w, lbw, ubw, g, lbg, ubg, X_objB, J

    def undock_objB_dockB(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.

        dock_id = 'B'
        obj_loc = self.dock_loc[dock_id]
        obj_loc[1] -= self.arm_length
        _, _, _, g_dock, lbg_dock, ubg_dock = self.add_dock_con(X, obj_loc)
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)

        return w, lbw, ubw, g, lbg, ubg, X_objB, J

    def undock_objB_dockC(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.

        dock_id = 'C'
        obj_loc = self.dock_loc[dock_id]
        obj_loc[1] -= self.arm_length
        _, _, _, g_dock, lbg_dock, ubg_dock = self.add_dock_con(X, obj_loc)
        g.extend(g_dock)
        lbg.extend(lbg_dock)
        ubg.extend(ubg_dock)

        return w, lbw, ubw, g, lbg, ubg, X_objB, J

    def grasp_objA(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objA = None
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X_objA, J

    def grasp_objB(self, X, U):
        w, lbw, ubw = [], [], []
        g, lbg, ubg = [], [], []

        X_objB = None
        J = 0.
        return w, lbw, ubw, g, lbg, ubg, X_objB, J

import pdb
import casadi
import numpy as np
import pybullet as pb

def update_f(Xprev, Uprev, params):
  _,_,_, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz = Xprev
  Fx,Fy,Fz, Mx,My,Mz = Uprev
  mass = params['mass']
  Jxx, Jyy, Jzz = np.diag(params['J']) 
  f = np.array([vx,
                vy,
                vz,
                0.5*(qy*wz - qz*wy + qw*wx),
                0.5*(qz*wx - qx*wz + qw*wy),
                0.5*(qx*wy - qy*wx + qw*wz),
                0.5*(-qx*wx - qy*wy - qz*wz),
                1/mass*Fx,
                1/mass*Fy,
                1/mass*Fz,
                (Mx + Jyy*wy*wz - Jzz*wy*wz)/Jxx,
                (My - Jxx*wx*wz + Jzz*wx*wz)/Jyy,
                (Mz + Jxx*wx*wy - Jyy*wx*wy)/Jzz
            ])
  return f

def update_A(Xprev, Uprev, params):
  _,_,_, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz = Xprev
  Fx,Fy,Fz, Mx,My,Mz = Uprev
  mass = params['mass']
  Jxx, Jyy, Jzz = np.diag(params['J']) 

  A = np.zeros((Xprev.size, Xprev.size))
  A[0:3, 7:10] = np.eye(3)

  # [0, w3/2, -w2/2, w1/2, q4/2, -q3/2, q2/2]
  A[3, 3:7] = 0.5*np.array([0., wz, -wy, wx])
  A[3, 10:13] = 0.5*np.array([qw, -qz, qy])

  # [-w3/2, 0, w1/2, w2/2, q3/2, q4/2, -q1/2]
  A[4, 3:7] = 0.5*np.array([-wz, 0., wx, wy])
  A[4, 10:13] = 0.5*np.array([qz, qw, -qx])

  # [w2/2, -w1/2, 0, w3/2, -q2/2, q1/2, q4/2]
  A[5, 3:7] = 0.5*np.array([wy, -wx, 0., wz])
  A[5, 10:13] = 0.5*np.array([qy, qx, qw])

  # [-w1/2, -w2/2, -w3/2, 0, -q1/2, -q2/2, -q3/2]
  A[6, 3:7] = 0.5*np.array([-wx, wy, wz, 0.])
  A[6, 10:13] = 0.5*np.array([-wz, -qy, -qz])

  # [0, 0, 0, 0, 0, (Jyy*w3 - Jzz*w3)/Jxx, (Jyy*w2 - Jzz*w2)/Jxx]
  A[10, 10:13] = np.array([0., (Jyy-Jzz)*wz/Jxx, (Jyy-Jzz)*wy/Jxx])

  # [0, 0, 0, 0, -(Jxx*w3 - Jzz*w3)/Jyy, 0, -(Jxx*w1 - Jzz*w1)/Jyy]
  A[11, 10:13] = np.array([(Jzz-Jxx)*wz/Jyy, 0., (Jzz-Jxx)*wx/Jyy])

  # [0, 0, 0, 0, (Jxx*w2 - Jyy*w2)/Jzz, (Jxx*w1 - Jyy*w1)/Jzz, 0]])
  A[12, 10:13] = np.array([(Jxx-Jyy)*wy/Jzz, (Jxx-Jyy)*wx/Jzz, 0.])
  return A

def update_B(Xprev, Uprev, params):
  _,_,_, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz = Xprev
  Fx,Fy,Fz, Mx,My,Mz = Uprev
  mass = params['mass']
  Jxx, Jyy, Jzz = np.diag(params['J']) 

  B = np.zeros((Xprev.size,Uprev.size))
  B[7:10, 0:3] = 1/mass * np.diag([Fx,Fy,Fz])
  B[10:13, 3:] = np.diag([1./Jxx, 1./Jyy, 1./Jzz])
  return B

def slerp(qi, qf, time):
    qi = qi / np.linalg.norm(qi, ord=2)
    qf = qf / np.linalg.norm(qf, ord=2)
    dot = qi.dot(qf)
    if dot < 0.:
        qi = -qi
        dot = -dot
    dot_thresh = 0.9995
    if dot > dot_thresh:
        # if quaternions are close
        out = qi + time*(qf-qi)
        out = out / np.linalg.norm(out, ord=2)
        return out
    theta_0 = np.arccos(dot)
    theta = theta_0 * time
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = s0*qi + s1*qf
    out = out / np.linalg.norm(out, ord=2)
    return out
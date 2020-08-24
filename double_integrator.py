import pdb
import casadi
import numpy as np
import pybullet as pb

def compute_Ak(Xprev, Uprev, params):
  rx,ry,vx,vz = Xprev
  Fx,Fy = Uprev

  mass = params['mass']
  Jxx, Jyy, Jzz = np.diag(params['J']) 
  dh = params['dh']

  n = Xprev.size
  Ak = np.eye(n)
  Ak[:int(n/2), int(n/2):] = dh*np.eye(int(n/2))

  return Ak 

def compute_Bk(Xprev, Uprev, params):
  rx,ry,vx,vz = Xprev
  Fx,Fy = Uprev

  mass = params['mass']
  Jxx, Jyy, Jzz = np.diag(params['J']) 
  dh = params['dh']

  n,m = Xprev.size, Uprev.size
  Bk = np.zeros((n,m))
  Bk[:int(n/2),:] = 0.5*dh**2*np.eye(int(n/2),m)
  Bk[int(n/2):,:] = dh*np.eye(int(n/2),m)
  Bk *= 1/mass

  return Bk 

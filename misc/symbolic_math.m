%% Rigid body dynamics
clear all; clc;
q = sym('q', [4,1], 'real');
w = sym('w', [3,1], 'real');
M = sym('M', [3,1], 'real');

% J = sym('J', [3,3], 'real');
syms Jxx Jyy Jzz Jxy Jxz Jyz
assume(Jxx, 'real');
assume(Jyy, 'real');
assume(Jzz, 'real');
%assume(Jxy, 'real');
%assume(Jyz, 'real');
%assume(Jxz, 'real');
%J = [Jxx,Jxy,Jxz;...
%    Jxy,Jyy,Jyz;...
%    Jxz,Jyz,Jzz];
J = diag([Jxx,Jyy,Jzz]);

omega_skew = [0, w(3),-w(2),w(1);...
            -w(3),0,w(1),w(2);...
            w(2),-w(1),0,w(3);...
            -w(1),-w(2),-w(3),0];
qdot = 0.5 * omega_skew * q;
wdot = inv(J)*(M - cross(w,J*w));

x = [q;w];
xdot = [qdot;wdot];
A = jacobian(xdot, x);
B = jacobian(xdot, M);

fid = fopen('a.txt', 'wt');
fprintf(fid, '%s\n', char(A));
fclose(fid);

fid = fopen('b.txt', 'wt');
fprintf(fid, '%s\n', char(B));
fclose(fid);

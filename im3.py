import random as rnd
import tkinter as tk
import time
from math import sin, cos, sqrt, exp
import numpy as np
import math
import sys

# Environmental Specification

num = 2 # number of agents
s = 6 # environment size in meters

# Agent Parameters (play with these)
k = 1.5
m = 2.0
t0 = 3
rad = .2  # Collision radius
sight = 7  # Neighbor search range
maxF = 5  # Maximum force/acceleration

pixelsize = 600
framedelay = 200
drawVels = True

maxIttr = 27
ittr = 0

win = tk.Tk()
canvas = tk.Canvas(win, width=pixelsize, height=pixelsize, background="#444")
canvas.pack()

# Initialized variables
ittr = 0
c = []  # center of agent
v = []  # velocity
gv = []  # goal velocity
nbr = []  # neighbor list
nd = []  # neighbor distance list
QUIT = False
paused = False
step = False

circles = []
velLines = []
gvLines = []

time_steps = 27
state_size = 4  # [x, y, vx, vy]
ensemble_size = 2000
Y = np.zeros((num, ensemble_size, state_size, time_steps))  # Actual state variables
noise_std = 0.1
state = np.load(r'C:\Users\iamma\OneDrive\Desktop\SURGE\data\states.npy')
times = np.load(r'C:\Users\iamma\OneDrive\Desktop\SURGE\data\times.npy')
for j in range(num):  # For each agent
    for t in range(time_steps):  # For each timestamp
        for i in range(ensemble_size):  # For each ensemble member
            Y[j, i, :, t] = state[j, :, t] + np.random.normal(0, noise_std, state_size)

X = np.zeros((num, ensemble_size, state_size, time_steps))
N = np.zeros((num, ensemble_size, state_size, time_steps))  # Ensemble of states
Z = np.zeros((num, ensemble_size, state_size, time_steps)) 
P = np.eye(state_size)  # State covariance matrix
Q = np.eye(state_size) * 0.01  # Process noise covariance matrix
Q = np.tile(np.expand_dims(Q, axis=0), (num, 1, 1))  # Expand Q for each agent
R = np.eye(state_size) * 0.05  # Measurement noise covariance matrix
agent1_positions = np.zeros((maxIttr, 2))
agent2_positions = np.zeros((maxIttr, 2))
entropy_values = []
times = []
def initSim():
    global rad
    print("")
    print("Simulation of Agents on a flat 2D torus.")
    print("Agents avoid collisions using principles based on the laws of anticipation seen in human pedestrians.")
    print("Agents are white circles, Red agent moves faster.")
    print("Green Arrow is Goal Velocity, Red Arrow is Current Velocity")
    print("SPACE to pause, 'S' to step frame-by-frame, 'V' to turn the velocity display on/off.")
    print("")
   
    for i in range(num):
        circles.append(canvas.create_oval(0, 0, rad, rad, fill="white"))
        velLines.append(canvas.create_line(0, 0, 10, 10, fill="red"))
        gvLines.append(canvas.create_line(0, 0, 10, 10, fill="green"))
        c.append(np.zeros(2))
        v.append(np.zeros(2))
        gv.append(np.zeros(2))
    
    # Set initial positions and velocities
    c[0][0] = 0
    c[1][0] = s - 1
    c[0][1] = s / 2
    c[1][1] = s / 2
    
    
    ang =  0
    en = 0
    v[0][0] = cos(ang)
    v[0][1] = sin(ang)
    gv[0] =2 * 1.5 * np.copy(v[0])
   
    canvas.itemconfig(circles[0], fill="#FAA")
    ang = (-1) * 3.141592
    v[1][0] = cos(ang)
    v[1][1] = sin(ang)
    gv[1] = 2 * 1.5 * np.copy(v[1])

    for j in range(ensemble_size):
        X[i, j, :, 0] = [c[i][0], c[i][1], v[i][0], v[i][1]]


def drawWorld():
    global rad, s
    for i in range(num):
        scale = pixelsize / s
        canvas.coords(circles[i], scale * (c[i][0] - rad), scale * (c[i][1] - rad), scale * (c[i][0] + rad),
                      scale * (c[i][1] + rad))
        canvas.coords(velLines[i], scale * c[i][0], scale * c[i][1], scale * (c[i][0] + 1. * rad * v[i][0]),
                      scale * (c[i][1] + 1. * rad * v[i][1]))
        canvas.coords(gvLines[i], scale * c[i][0], scale * c[i][1], scale * (c[i][0] + 1. * rad * gv[i][0]),
                      scale * (c[i][1] + 1. * rad * gv[i][1]))
        if drawVels:
            canvas.itemconfigure(velLines[i], state="normal")
            canvas.itemconfigure(gvLines[i], state="normal")
        else:
            canvas.itemconfigure(velLines[i], state="hidden")
            canvas.itemconfigure(gvLines[i], state="hidden")
        double = False
        newX = c[i][0]
        newY = c[i][1]
        if c[i][0] < rad:
            newX += s
            double = True
        if c[i][0] > s - rad:
            newX -= s
            double = True
        if c[i][1] < rad:
            newY += s
            double = True
        if c[i][1] > s - rad:
            newY -= s
            double = True
        if double:
            pass


def findNeighbors():
    global nbr, nd, c
    nbr = []
    nd = []
    for i in range(num):
        nbr.append([])
        nd.append([])
        for j in range(num):
            if i == j:
                continue
            d = c[i] - c[j]
            if d[0] > s / 2.:
                d[0] = s - d[0]
            if d[1] > s / 2.:
                d[1] = s - d[1]
            if d[0] < -s / 2.:
                d[0] = d[0] + s
            if d[1] < -s / 2.:
                d[1] = d[1] + s
            l2 = d.dot(d)
            s2 = sight ** 2
            if l2 < s2:
                nbr[i].append(j)
                nd[i].append(sqrt(l2))

def E(t):
    return (B / t ** m) * exp(-t / t0)

def rdiff(pa, pb, va, vb, ra, rb):
    p = pb - pa  # relative position
    return (sqrt(p.dot(p)))

def ttc(pa, pb, va, vb, ra, rb):
    maxt = 999
    p = pb - pa  # relative position
    if p[0] > s / 2.:
        p[0] = p[0] - s
    if p[1] > s / 2.:
        p[1] = p[1] - s
    if p[0] < -s / 2.:
        p[0] = p[0] + s
    if p[1] < -s / 2.:
        p[1] = p[1] + s
    rv = vb - va  # relative velocity
    a = rv.dot(rv)
    b = 2 * rv.dot(p)
    c = p.dot(p) - (ra + rb) ** 2
    det = b * b - 4 * a * c
    t1 = maxt
    t2 = maxt
    if det > 0:
        t1 = (-b + sqrt(det)) / (2 * a)
        t2 = (-b - sqrt(det)) / (2 * a)
    t = min(t1, t2)
    if t < 0 and max(t1, t2) > 0:  # we are colliding
        t = 100
    if t < 0:
        t = maxt
    if t > maxt:
        t = maxt
    return t

def dE(pa, pb, va, vb, ra, rb):
    global k, m, t0
    INFTY = 999
    maxt = 999
    w = pb - pa
    if w[0] > s / 2.:
        w[0] = w[0] - s  # wrap around for torus
    if w[1] > s / 2.:
        w[1] = w[1] - s
    if w[0] < -s / 2.:
        w[0] = w[0] + s
    if w[1] < -s / 2.:
        w[1] = w[1] + s
    v = va - vb
    radius = ra + rb
    dist = sqrt(w[0] ** 2 + w[1] ** 2)
    if radius > dist:
        radius = .99 * dist
    a = v.dot(v)
    b = w.dot(v)
    c = w.dot(w) - radius * radius
    discr = b * b - a * c
    if (discr < 0) or (a < 0.001 and a > - 0.001):
        return np.array([0, 0])
    discr = sqrt(discr)
    t1 = (b - discr) / a
    t = t1
    if (t < 0):
        return np.array([0, 0])
    if (t > maxt):
        return np.array([0, 0])
    d = k * exp(-t / t0) * (v - (v * b - w * a) / (discr)) / (a * t ** m) * (m / t + 1 / t0)
    return d


def update(dt):
    global c
    findNeighbors()
    F = []  # force
    for i in range(num):
        F.append(np.zeros(2))
    for i in range(num):
        F[i] += (gv[i] - v[i]) / .5
        F[i] += 1 * np.array([rnd.uniform(-1., 1.), rnd.uniform(-1., 1.)])
        for n, j in enumerate(nbr[i]): 
            t = ttc(c[i], c[j], v[i], v[j], rad, rad)
            d = c[i] - c[j]
            if d[0] > s / 2.:
                d[0] = d[0] - s  
            if d[1] > s / 2.:
                d[1] = d[1] - s
            if d[0] < -s / 2.:
                d[0] = d[0] + s
            if d[1] < -s / 2.:
                d[1] = d[1] + s
            r = rad
            dist = sqrt(d.dot(d))
            if dist < 2 * rad:
                r = dist / 2.001 
            dEdx = dE(c[i], c[j], v[i], v[j], r, r)
            FAvoid = -dEdx
            mag = np.sqrt(FAvoid.dot(FAvoid))
            if (mag > maxF):
                FAvoid = maxF * FAvoid / mag
            F[i] += FAvoid
        
    for i in range(num):
        a = F[i]
        v[i] += a * dt
        c[i] += v[i] * dt
        if c[i][0] < 0:
            c[i][0] = s 
        if c[i][1] < 0:
            c[i][1] = s
        if c[i][0] > s:
            c[i][0] = 0
        if c[i][1] > s:
            c[i][1] = 0
    agent1_positions[ittr] = c[0]
    agent2_positions[ittr] = c[1]

def entropy():
    global Q, X, Z, Y, N, num, ensemble_size, state_size

    tp = num

    for j in range(num):
        N[j, 0, :, :] = X[j, 0, :, :]
        for i in range(ensemble_size - 1):
            N[j, i + 1, 0, ittr] = X[j, i, 0, ittr] + rnd.gauss(0, Q[j, 0, 0])
            N[j, i + 1, 1, ittr] = X[j, i, 1, ittr] + rnd.gauss(0, Q[j, 1, 1])
            N[j, i + 1, 2, ittr] = X[j, i, 2, ittr] + rnd.gauss(0, Q[j, 2, 2])
            N[j, i + 1, 3, ittr] = X[j, i, 3, ittr] + rnd.gauss(0, Q[j, 3, 3])

            Z[j, i, 0, ittr] = rnd.gauss(0, R[0, 0])
            Z[j, i, 1, ittr] = rnd.gauss(0, R[1, 1])
            Z[j, i, 2, ittr] = X[j, i, 2, ittr] + rnd.gauss(0, R[2, 2])
            Z[j, i, 3, ittr] = X[j, i, 3, ittr] + rnd.gauss(0, R[3, 3])

        z_bar_k = np.mean(Z[j, :, :, ittr], axis=0)
        Z_k = np.cov(Z[j, :, :, ittr], rowvar=False)

        if np.linalg.cond(Z_k) > 1 / sys.float_info.epsilon:
            Z_k += 1e-6 * np.eye(state_size)  

        A_hat_k = np.mean(N[j, :, :, ittr], axis=0)

        cov_i = np.zeros((state_size, state_size))
        for k in range(ensemble_size):
            cov_i += np.outer((N[j, k, :, ittr] - A_hat_k), (Z[j, k, :, ittr] - z_bar_k))

        cov_i /= ensemble_size

        for k in range(ensemble_size):
            N[j, k, :, ittr] += cov_i @ np.linalg.inv(Z_k) @ (Y[j, k, :, ittr] - z_bar_k)

    for k in range(tp - 1):
        for i in range(ensemble_size):
            diff = N[k, i, :, :] - X[k, i, :, :]
            Q[k] += (1 / ensemble_size) * (diff @ diff.T)

    w = tp // 5
    for k in range(tp):
        for I in range(max(0, k - w), min(tp, k + w) + 1):
            Q[k] += Q[I]
        Q[k] /= (2 * w + 1)

    Q_ = np.mean(Q, axis=0)
    det_Q = np.linalg.det(Q_)
    return Q



def calculate_entropy():
    global Q
    Q_ = np.mean(Q, axis=0)
    det_Q = np.linalg.det(Q_)
    entropy = (1 / 2) * (np.log((2 * np.pi * np.e) ** 4) + np.log(det_Q))
    

    return entropy


def on_key_press(event):
    global paused, step, QUIT, drawVels
    if event.keysym == "space":
        paused = not paused
    if event.keysym == "s":
        step = True
        paused = False
    if event.keysym == "v":
        drawVels = not drawVels
    if event.keysym == "Escape":
        QUIT = True
    
def drawFrame(dt=0.05):
    global start_time, step, paused, ittr, en, Q

    if ittr >= maxIttr or QUIT:  
        np.save('agent1_positions.npy', agent1_positions)
        np.save('agent2_positions.npy', agent2_positions)
        np.save('entropy',np.array(entropy_values))
        np.save('times',np.array(times))
        print("%s iterations ran ... quitting" % ittr)
        win.destroy()
    else:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        if not paused:
            update(0.12)
            Q = entropy()
            en = calculate_entropy()
            # if math.isinf(en):
            #     Q = np.eye(state_size) * 0.1  

            #     Q = np.tile(np.expand_dims(Q, axis=0), (num, 1, 1))  
            current_time = ittr*framedelay
            print(f"Entropy --> {en}")
            entropy_values.append(en)
            times.append(current_time)
            ittr += 1
        drawWorld()
        if step:
            step = False
            paused = True
        win.title("K.S.G. 2014 (Under Review)")
        win.after(framedelay, drawFrame)



win.bind("<space>", on_key_press)
win.bind("s", on_key_press)
win.bind("<Escape>", on_key_press)
win.bind("v", on_key_press)
print()
initSim()


start_time = time.time()
win.after(framedelay, drawFrame)
win.mainloop()

import numpy as np
import pygame
from numpy.linalg import inv
import random
import matplotlib.pyplot as plt
import sympy as sy
from filterpy.stats import plot_covariance
import math
plt.style.use('ggplot')



DIAMETER = 80
DT = 0.01

# Define some colors
BLACK = (0 ,0, 0)
WHITE = (255, 255, 255)

class Robot():
    def __init__(self):
        self.length = 600
        self.width = 600
        self.radius = 40
        self.H = np.eye(3)
        self.P = np.eye(3)
        self.max_motor_spd = np.pi
        self.motor_spd = 0.05 * self.max_motor_spd
        self.laser = 10
        self.imu = 0.25
        self.wheels_d = 85
        self.rr = 0.5
        self.noise = np.diag([self.laser ** 2, self.laser ** 2, self.imu ** 2])
        x_pos, y_pos, theta, l_spd, r_spd, radius, d, t = sy.symbols('x_pos, y_pos, theta, l_spd, r_spd, radius, d, t')
        self.func = self.symbolize(x_pos, y_pos, theta, l_spd, r_spd, radius, d, t)
        self.state_dev = self.func.jacobian(sy.Matrix([x_pos, y_pos, theta]))
        self.input_dev = self.func.jacobian(sy.Matrix([l_spd, r_spd]))

        # define the initial state of the robot.
        self.cur_state = np.array([200, 200 ,np.pi/2])
        self.theta = self.cur_state[2]
        self.speed = np.array([0, 0])
        self.l_spd = l_spd
        self.r_spd = r_spd
        self.t = 0
        self.dic = self.init_dic(x_pos, y_pos, theta, l_spd, r_spd, radius, d, t)




    def symbolize(self,x_pos, y_pos, theta, l_spd, r_spd, radius, d, t):
        func = sy.Matrix([[x_pos+1/2*sy.cos(theta)*(l_spd+r_spd)*radius*t], [y_pos+1/2*sy.sin(theta)*(l_spd+r_spd)*radius*t],
                          [theta+(r_spd-l_spd)*radius/d*t]])
        return func

    def init_dic(self, x_pos, y_pos, theta, l_spd, r_spd, radius, d, t):
        dic = {}
        dic[x_pos] = self.cur_state[0]
        dic[y_pos] = self.cur_state[1]
        dic[theta] = self.cur_state[2]
        dic[l_spd] = self.speed[0]
        dic[r_spd] = self.speed[1]
        dic[radius] = self.radius
        dic[d] = self.wheels_d
        dic[t] = self.rr
        return dic

    def new_state(self, cur_state, spd, rr):

        keys = pygame.key.get_pressed()
        if keys[ord("w")]:
            self.l_spd += 0.001
        if keys[ord("s")]:
            self.l_spd -= 0.001
        if keys[ord("i")]:
            self.r_spd += 0.001
        if keys[ord("k")]:
            self.r_spd -= 0.001

        l_spd, r_spd = spd[0], spd[1]
        theta = cur_state[2]
        v = 0.5 * self.radius * (l_spd + r_spd)
        l_dis = rr * l_spd
        r_dis = rr * r_spd
        #update = np.array((sy.cos(theta)*v*rr, sy.sin(theta)*v*rr, (r_dis-l_dis)/1.5))
        update = np.array((sy.sin(theta)*v*rr, -sy.cos(theta)*v*rr, (l_dis-r_dis)/1.5))
        new_state = cur_state + update
        return new_state


    def res(self, msr):
        res = msr - self.H.dot(self.cur_state)
        res[2] = res[2] % (2*np.pi)
        if res[2] > np.pi:
            res[2] = res[2] - 2 * np.pi
        return res

    def update(self, res):
        K1 = np.dot(self.P, self.H.T)
        K2 = inv(np.dot(self.H, self.P).dot(self.H.T) + self.noise)
        self.K = K1.dot(K2)
        self.cur_state = self.cur_state + self.K.dot(res)
        self.P = (np.eye(3) - self.K.dot(self.H)).dot(self.P)

    def pred(self, speed):
        self.cur_state = self.new_state(self.cur_state, speed, self.rr)
        self.dic[self.l_spd] = speed[0]
        self.dic[self.r_spd] = speed[1]
        self.dic[self.theta] = self.cur_state[2]
        state_dev = np.array(self.state_dev.evalf(subs=self.dic)).astype(float)
        input_dev = np.array(self.input_dev.evalf(subs=self.dic)).astype(float)
        # diagonal matrix
        M = np.array([[self.motor_spd**2, 0], [0, self.motor_spd**2]])
        self.P = np.dot(state_dev, self.P).dot(state_dev.T) + np.dot(input_dev, M).dot(input_dev.T)
        self.t = self.t + 1

    def add_noise(self, cur_state):
        # rand = np.array([[random.uniform(0, 4), random.uniform(0, 4), random.uniform(0, 4)]])
        state_with_noise = cur_state + np.sqrt(self.noise).dot(np.random.randn(3))
        return state_with_noise


def main():
    pygame.init()

    lims = 600
    screen = pygame.display.set_mode([600, 600],0)


    # configuration space of size 600x600. All zeros, no landmarks
    space = np.zeros((600, 600))

    # declare robot at initial position with initial heading.
    myrobot = Robot()

    #movingsprites = pygame.sprite.Group()
    #movingsprites.add(myrobot)

    draw = True
    flag = True
  
    
    screen.fill(BLACK)


    landmarks = np.array([[300,300]])

    xs = []
    robot_x = []
    robot_y = []
    mu_x = []
    mu_y = []

    # the initial speed of two wheels.
    spd = np.array((0.1, 0.1))
    myrobot.l_spd = spd[0]
    myrobot.r_spd = spd[1]

    while(flag):
        if (draw):
            screen.fill(BLACK)
            img = np.zeros((600,600,3), np.uint8)
            
            # Draw Circle (Robot)
            pygame.draw.circle(screen, (0,0,255), (int(round(myrobot.cur_state[0])), int(round(myrobot.cur_state[1]))), int(DIAMETER/2))

            # Draw line to show heading direction (theta)
            radius = DIAMETER/2
            mylinex = myrobot.cur_state[0] + radius*math.sin(myrobot.cur_state[2])
            myliney = myrobot.cur_state[1] - radius*math.cos(myrobot.cur_state[2])

            pygame.draw.line(screen,WHITE,(int(round(myrobot.cur_state[0])), int(round(myrobot.cur_state[1]))), (int(round(mylinex)), int(round(myliney))), 2)
            
            pygame.display.flip()
        
    #     # move the robot
    #     ## action is a tuple based on user input. (speed, turn). speed is either 1 or 0.
    #     ## turn is -1,0, or 1. -1 is counterclockwise. 1 is clockwise, 0 is no turn.
    #     pos_prev = myrobot.pos
    #     theta_prev = myrobot.theta

        # move the robot
        real_spd = np.array([myrobot.l_spd,myrobot.r_spd]) + myrobot.motor_spd * np.random.randn(2)
        myrobot.cur_state = myrobot.new_state(myrobot.cur_state, real_spd, DT)
        print((myrobot.l_spd, myrobot.r_spd))



    #     action = (speed, turn)
    #     action = myrobot.updateState(action)

    #     #print(action)
    #     robot_pos = np.array(myrobot.pos)

    #     # observations based on landmarks
    #     zs = (norm(landmarks - robot_pos, axis=1) + randn(NL)*1.1)

    #     # predict based on actions
    #     PF.predict(particles, (myrobot.pos[0] - pos_prev[0], myrobot.pos[1] - pos_prev[1], math.radians(myrobot.theta - theta_prev)), std=(0.25, 1), dt=DT)

    #     PF.update(particles, weights, z=zs, R=1.1, landmarks=landmarks)

    #     if PF.neff(weights) < N/2:
    #         indexes = systematic_resample(weights)
    #         PF.resample_from_index(particles, weights, indexes)
    #         assert np.allclose(weights, 1/N)

    #     mu, var = PF.estimate(particles, weights)
    #     xs.append(mu)

    #     mu_x.append(mu[0])
    #     mu_y.append(mu[1])
    #     robot_x.append(robot_pos[0])
    #     robot_y.append(robot_pos[1])

        
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN):
                if (event.key == ord('q')):
                    flag = False
                #if (event.key == ord('s')):
                 #   speed = ~speed
    # robot_x = np.asarray(robot_x)
    # robot_y = np.asarray(robot_y)
    # mu_x = np.asarray(mu_x)
    # mu_y = np.asarray(mu_y)

    # print()

    # plt.figure()
    # plt.plot(robot_x, robot_y, color='k')
    # plt.plot(mu_x, mu_y, color='r')
    # plt.show()


main()
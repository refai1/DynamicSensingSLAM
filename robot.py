## kinematic and sensor model for our robot in a given space.

import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from numpy.random import uniform
import pygame
import math
import particleFilter.pf as PF


from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats

DIAMETER = 80
DT = 0.5

# Define some colors
BLACK = (0 ,0, 0)
WHITE = (255, 255, 255)

class Robot:
	"""docstring for Robot"""
	def __init__(self, pos, theta):
		self.pos = pos
		self.theta = theta

	# This function takes in the robot's action (different wheel velocities)
	# and allows user input to change the velocity. moves the robot and updates the state.
	# it returns the action at the end of the call to be used on the next call.
	def updateState(self, action):
		l_spd = action[0]
		r_spd = action[1]

		keys = pygame.key.get_pressed()
		if keys[ord("w")]:
			l_spd += 0.001
		if keys[ord("s")]:
			l_spd -= 0.001
		if keys[ord("i")]:
			r_spd += 0.001
		if keys[ord("k")]:
			r_spd -= 0.001

		#l_spd, r_spd = [0], spd[1]

		theta = self.theta
		v = 0.5 * DIAMETER*0.5 * (l_spd + r_spd)
		l_dis = DT * l_spd
		r_dis = DT * r_spd
		#update = np.array((sy.cos(theta)*v*rr, sy.sin(theta)*v*rr, (r_dis-l_dis)/1.5))
		update = np.array((math.sin(theta)*v*DT, -math.cos(theta)*v*DT, (l_dis-r_dis)/1.5))
		#new_state = cur_state + update
		
		newx = self.pos[0] + update[0]# + (0.2*(np.random.random_sample()-0.5)*update[0])
		newy = self.pos[1] + update[1] #+ (0.2*(np.random.random_sample()-0.5)*update[1])
		newtheta = self.theta + update[2]

		self.pos = (newx, newy)
		self.theta = newtheta

		return (l_spd, r_spd)





def main():

	# set up game/animation screen
	pygame.init()
	screen = pygame.display.set_mode([600, 600],0)
	screen.fill(BLACK)


	# configuration space of size 600x600. All zeros, no landmarks
	space = np.zeros((600, 600))

	# declare robot at initial position with initial heading.
	myrobot = Robot((50,50), math.radians(180))

	# action is the velocity of the left and right wheel respectively. 
	action = (0,0)

	

	# define a single landmark in the middle. N=5000 particles
	landmarks = np.array([[300,300]])
	NL = len(landmarks)
	N = 5000

	# initially we have a good idea of where the robot is. weights are uniform in that space
	particles = PF.create_gaussian_particles(mean=[myrobot.pos[0],myrobot.pos[1],math.radians(myrobot.theta)], std=(20, 20, np.pi/4), N=N)
	weights = np.ones(N)/N

	#empty arrays for plotting
	xs = []
	robot_x = []
	robot_y = []
	mu_x = []
	mu_y = []
	err = []

	iteration = 0
	draw = True
	flag = True
	while(flag):
		if (draw):
			screen.fill(BLACK)
			img = np.zeros((600,600,3), np.uint8)
			
			# Draw Circle (Robot)
			pygame.draw.circle(screen, (0,0,255), (int(round(myrobot.pos[0])), int(round(myrobot.pos[1]))), int(DIAMETER/2))

			# Draw line to show heading direction (theta)
			radius = DIAMETER/2
			mylinex = myrobot.pos[0] + radius*math.sin(myrobot.theta)
			myliney = myrobot.pos[1] - radius*math.cos(myrobot.theta)

			pygame.draw.line(screen,WHITE,(int(round(myrobot.pos[0])), int(round(myrobot.pos[1]))), (int(round(mylinex)), int(round(myliney))), 2)
			
			pygame.display.flip()
		
		
		pos_prev = myrobot.pos
		theta_prev = myrobot.theta

		if(iteration < 50):
			action = (0,0)
		elif(iteration < 400):
			action = (0.02, 0.02)
		elif(iteration < 800):
			action = (0.01, 0.02)
		elif(iteration < 1200):
			action = (0.02, 0.02)
		elif(iteration < 1600):
			action = (0.02, 0.01)
		elif(iteration < 1900):
			action = (0.02, 0.02)
		else:
			action = (0,0)
		# move the robot	
		action = myrobot.updateState(action)

		#print(action)
		robot_pos = np.array(myrobot.pos)

		# observations based on landmarks
		zs = (norm(landmarks - robot_pos, axis=1) + randn(NL)*1.1)

		# predict based on actions
		PF.predict(particles, action, std=(0.02, 0.1), dt=DT)

		PF.update(particles, weights, z=zs, R=1.1, landmarks=landmarks)

		if PF.neff(weights) < N/2:
			indexes = systematic_resample(weights)
			PF.resample_from_index(particles, weights, indexes)
			assert np.allclose(weights, 1/N)

		mu, var = PF.estimate(particles, weights)
		xs.append(mu)

		mu_x.append(mu[0])
		mu_y.append(mu[1])
		robot_x.append(robot_pos[0])
		robot_y.append(robot_pos[1])

		err.append(math.sqrt((robot_pos[0] - mu[0])**2 + (robot_pos[1] - mu[1])**2))

		iteration += 1
		for event in pygame.event.get():
			if (event.type == pygame.KEYDOWN):
				if (event.key == ord('q')):
					flag = False
				#if (event.key == ord('s')):
				#	speed = ~speed
	robot_x = np.asarray(robot_x)
	robot_y = np.asarray(robot_y)
	mu_x = np.asarray(mu_x)
	mu_y = np.asarray(mu_y)

	print()

	plt.figure()
	plt.plot(robot_x, robot_y, color='k')
	plt.plot(mu_x, mu_y, color='r')

	plt.figure(2)
	plt.plot(err, color='k')
	print(sum(err)/iteration)
	
	plt.show()


main()
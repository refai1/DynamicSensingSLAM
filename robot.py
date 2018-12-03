## kinematic and sensor model for our robot in a given space.

import numpy as np 
import matplotlib.pyplot as plt
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


	def updateState(self, action):
		speed = action[0]
		turn = action[1]
		
		keys = pygame.key.get_pressed()
		if keys[pygame.K_LEFT]:
			turn = turn - 1
			if (turn < -1):
				turn = -1
		if keys[pygame.K_RIGHT]:
			turn = turn + 1
			if (turn > 1):
				turn = 1

		newtheta = self.theta + turn*math.radians(1)
		newx = self.pos[0] - speed*DT*math.sin(self.theta)
		newy = self.pos[1] + speed*DT*math.cos(self.theta)

		self.pos = (newx,newy)
		self.theta = newtheta
		return (speed,0)





def main():


	pygame.init()

	lims = 600
	screen = pygame.display.set_mode([600, 600],0)


	# configuration space of size 600x600. All zeros, no landmarks
	space = np.zeros((600, 600))

	# declare robot at initial position with initial heading.
	myrobot = Robot((50,50), math.radians(180))

	#movingsprites = pygame.sprite.Group()
	#movingsprites.add(myrobot)

	draw = True
	flag = True
	# the user can press s to toggle speed between 0 and 1
	# the user can press -> or <- to toggle turn clockwise or counter respectively.
	speed = 0
	turn = 0

	action = (speed,turn)

	screen.fill(BLACK)


	landmarks = np.array([[300,300]])
	NL = len(landmarks)
	N = 5000
	#plt.figure()

	# initially we have a good idea of where the robot is
	particles = PF.create_gaussian_particles(mean=[myrobot.pos[0],myrobot.pos[1],math.radians(myrobot.theta)], std=(20, 20, np.pi/4), N=N)
	weights = np.ones(N)/N

	xs = []
	#robot_pos = np.array(myrobot.pos)

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
		
		# move the robot
		## action is a tuple based on user input. (speed, turn). speed is either 1 or 0.
		## turn is -1,0, or 1. -1 is counterclockwise. 1 is clockwise, 0 is no turn.
		pos_prev = myrobot.pos
		theta_prev = myrobot.theta

		action = (speed, turn)
		action = myrobot.updateState(action)

		#print(action)
		robot_pos = np.array(myrobot.pos)

		# observations based on landmarks
		zs = (norm(landmarks - robot_pos, axis=1) + randn(NL)*1.1)

		# predict based on actions
		PF.predict(particles, (myrobot.pos[0] - pos_prev[0], myrobot.pos[1] - pos_prev[1], math.radians(myrobot.theta - theta_prev)), std=(0.25, 1), dt=DT)

		PF.update(particles, weights, z=zs, R=1.1, landmarks=landmarks)

		if PF.neff(weights) < N/2:
			indexes = systematic_resample(weights)
			PF.resample_from_index(particles, weights, indexes)
			assert np.allclose(weights, 1/N)

		mu, var = PF.estimate(particles, weights)
		xs.append(mu)

		print("actual", robot_pos)
		print("predicted", mu)
		p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+', color='k', s=180, lw=3)
		p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')


		for event in pygame.event.get():
			if (event.type == pygame.KEYDOWN):
				if (event.key == ord('q')):
					flag = False
				if (event.key == ord('s')):
					speed = ~speed
	#plt.xlim(lims)
	#plt.ylim(lims)
	#plt.show()
		
main()
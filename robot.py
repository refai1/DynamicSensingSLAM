## kinematic and sensor model for our robot in a given space.

import numpy as np 
#import cv2
import pygame
import math

DIAMETER = 80
DT = 0.05

# Define some colors
BLACK = (0 ,0, 0)
WHITE = (255, 255, 255)

class Robot:
	"""docstring for Robot"""
	def __init__(self, pos, theta):
		self.pos = pos
		self.theta = theta


	def update(self, action):
		speed = action[0]
		turn = action[1]
		
		for event in pygame.event.get():
			if (event.type == pygame.KEYDOWN):
				if (event.key == ord('s')):
					speed = ~speed

		keys = pygame.key.get_pressed()
		if keys[pygame.K_LEFT]:
			turn = turn - 1
			if (turn < -1):
				turn = -1
		if keys[pygame.K_RIGHT]:
			turn = turn + 1
			if (turn > 1):
				turn = 1

		newtheta = self.theta + turn*math.radians(0.1)
		newx = self.pos[0] - speed*DT*math.sin(self.theta)
		newy = self.pos[1] + speed*DT*math.cos(self.theta)

		self.pos = (newx,newy)
		self.theta = newtheta
		return (speed,0)







def main():

	pygame.init()

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
			#cv2.line(img, (int(round(myrobot.pos[0])), int(round(myrobot.pos[1]))), (int(round(mylinex)), int(round(myliney))), (0,0,255), 2)
			pygame.draw.line(screen,WHITE,(int(round(myrobot.pos[0])), int(round(myrobot.pos[1]))), (int(round(mylinex)), int(round(myliney))), 2)
			#pygame.draw.line(screen,WHITE,True, ((300,0), (300,100)), 2)
			
			#screen.blit(frame, (0,0))
			pygame.display.flip()
		
		# move the robot
		## action is a tuple based on user input. (speed, turn). speed is either 1 or 0.
		## turn is -1,0, or 1. -1 is counterclockwise. 1 is clockwise, 0 is no turn.
		action = myrobot.update(action)

		for event in pygame.event.get():
			if (event.type == pygame.KEYDOWN):
				if (event.key == ord('q')):
					flag = False
		
main()
import math

import cv2
import numpy as np


"""
BumpSlam
"""

def draw_box(box, canvas, color, thickness):
    box2 = cv2.boxPoints(box)
    box3 = np.int0(box2)
    cv2.drawContours(canvas, [box3], 0, color, thickness)


def calc_roi_and_mask_circle(center, radius, img):
    center = np.int0(center)
    roi = img[center[1] - radius:center[1] + radius, center[0] - radius:center[0] + radius]
    mask = np.full_like(roi, 0)

    cv2.circle(mask, (radius, radius), radius, 1.0, -1)

    return roi, mask


def calc_roi_and_mask(points, img):
    bb = cv2.boundingRect(points)
    roi = img[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
    mask = np.full_like(roi, 0)

    adjusted_box = points - (bb[0], bb[1])

    cv2.drawContours(mask, [np.int0(adjusted_box)], 0, 1.0, -1)

    return roi, mask


def sus(agents, number_to_keep):
    total = sum(a.fitness for a in agents)
    assert total > 0
    dist = total / number_to_keep
    start = np.random.uniform(0.0, dist)
    pointers = [start + i * dist for i in range(number_to_keep)]

    keep = []
    for p in pointers:
        i = 0
        so_far = agents[0].fitness
        while so_far < p:
            i += 1
            so_far += agents[i].fitness
        new_agent = agents[i].clone()
        new_agent.reset()
        keep.append(new_agent)

    return keep


class Agent:
    def __init__(self, x, y, angle, width=25, height=40, speed=5, fitness=1):
        self.position = np.array([x, y])
        self.heading = angle
        self.width = width
        self.height = height
        self.speed = speed
        self.fitness = fitness

    def get_box(self):
        return self.position, (self.height, self.width), np.degrees(self.heading)

    def get_front_corners(self):
        points = cv2.boxPoints(self.get_box())
        return points[2:]

    def draw_box(self, canvas):
        box = self.get_box()
        draw_box(box, canvas, (0, 191, 255), 1)
        points = self.get_front_corners()

        cv2.circle(canvas, tuple(points[0]), 5, (0, 0, 255), -1)
        cv2.circle(canvas, tuple(points[1]), 5, (0, 0, 255), -1)

    def draw_point(self, canvas):
        cv2.circle(canvas, tuple(np.int0(self.position)), 2, (255, 255, 255), -1)

    def move(self):
        self.position += self.speed * np.array([math.cos(self.heading), math.sin(self.heading)])

    def rotate(self, angle_offset):
        self.heading = (self.heading + angle_offset) % (2.0 * math.pi)

    def reverse(self):
        self.speed = -self.speed

    def set_speed(self, speed):
        self.speed = speed

    def clone(self):
        return Agent(self.position[0], self.position[1], self.heading, self.width, self.heading,
                     self.speed, self.fitness)

    def reset(self):
        self.fitness = 1
        self.rotate(np.random.uniform(np.radians(-2), np.radians(2)))

    def reduce_fitness(self):
        self.fitness = 0


class Environment:
    def __init__(self):
        self.width = 768
        self.height = 512

        self.cells = np.zeros((self.height, self.width))
        cv2.rectangle(self.cells, (20, 20), (self.width - 20, self.height - 20), 1.0)
        cv2.circle(self.cells, (150, 250), 50, 1.0)

    def draw(self, canvas):
        color_img = cv2.cvtColor((255.0 * self.cells).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        canvas[:] = np.maximum(color_img, canvas)

    def get(self, points):
        roi, mask = calc_roi_and_mask(points, self.cells)

        return np.sum(np.multiply(roi, mask))

    def get_circle(self, center, radius):
        roi, mask = calc_roi_and_mask_circle(center, radius, self.cells)

        return np.sum(np.multiply(roi, mask))


class Estimation:
    def __init__(self):
        self.width = 768
        self.height = 512

        self.cells = 0.5*np.ones((self.height, self.width))

    def draw(self, canvas):
        color_img = cv2.cvtColor((255.0 * self.cells).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        canvas[:] = np.maximum(color_img, canvas)

    def get(self, points):
        roi, mask = calc_roi_and_mask(points, self.cells)

        return np.sum(np.multiply(roi, mask))

    def set(self, box, value):
        pass

    def increase(self, box, amount):
        roi, mask = calc_roi_and_mask(box, self.cells)

        roi += mask * amount
        roi[:] = np.clip(roi, 0.0, 1.0)

    def decrease(self, box, amount):
        roi, mask = calc_roi_and_mask(box, self.cells)

        roi -= mask * amount
        roi[:] = np.clip(roi, 0.0, 1.0)


class Simulation:
    def __init__(self):
        self.num_hypotheses = 1000

        self.img = np.ones((512, 768, 3), np.uint8)

        self.env = Environment()
        self.est = Estimation()

        self.pilot = Agent(768 / 2, 512 / 2, np.random.uniform(0, 2 * np.pi))

        self.hypotheses = [Agent(768 / 2, 512 / 2, np.random.uniform(0, 2*np.pi)) for _ in range(self.num_hypotheses)]

    def move_and_get_contour(self, agent):
        last_corners = agent.get_front_corners()
        agent.move()
        new_corners = agent.get_front_corners()

        return np.concatenate((last_corners, new_corners[::-1]))

    def is_collision(self, contour):
        return self.env.get(contour) > 1

    def is_collision_circle(self, center, radius):
        return self.env.get_circle(center, radius) > 1

    def random_angle(self):
        return np.random.uniform(np.radians(135), np.radians(135 + 90))

    def turn_around(self, agent, angle):
        agent.reverse()
        for r in range(int(25.0 / abs(agent.speed))):
            agent.move()
        agent.rotate(angle)
        agent.reverse()

    def handle_hypothesis_collision(self, agent, contour, angle):
        self.est.increase(contour, 0.02)
        self.turn_around(agent, angle)

    def handle_hypothesis_miss(self, agent, contour):
        return self.est.decrease(contour, 0.02)

    def run(self):
        while True:
            self.img.fill(0)

            contour = self.move_and_get_contour(self.pilot)
            if self.is_collision(contour):
                angle = self.random_angle()
                self.turn_around(self.pilot, angle)

                for agent in self.hypotheses:
                    contour = self.move_and_get_contour(agent)
                    self.handle_hypothesis_collision(agent, contour, angle)

                    if not self.is_collision_circle(agent.position, 50):
                        agent.reduce_fitness()

            else:
                for agent in self.hypotheses:
                    contour = self.move_and_get_contour(agent)
                    self.handle_hypothesis_miss(agent, contour)

                    if self.is_collision_circle(agent.position, 20):
                        agent.reduce_fitness()

            self.hypotheses = sus(self.hypotheses, self.num_hypotheses)

            for agent in self.hypotheses:
                agent.draw_point(self.img)

            self.pilot.draw_box(self.img)
            self.env.draw(self.img)
            #self.est.draw(self.img)

            cv2.imshow('Draw01', self.img)
            cv2.waitKey(1)

def main():
    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
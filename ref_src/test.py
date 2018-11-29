from unittest import TestCase
from sim import sus

class TestSus(TestCase):
    def test_sus(self):
        class FakeAgent:
            def __init__(self, fitness):
                self.fitness = fitness

            def clone_and_reset(self):
                return self

            def __str__(self):
                return str(self.fitness)

            def __repr__(self):
                return "FakeAgent(%d)" % self.fitness

        agents = [FakeAgent(100), FakeAgent(50)]

        result = sus(agents, 5)
        self.assertEqual(result[0].fitness, 100)
        self.assertEqual(result[1].fitness, 100)
        self.assertEqual(result[4].fitness, 50)
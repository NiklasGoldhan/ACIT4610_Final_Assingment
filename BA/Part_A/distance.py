from math import floor, sqrt

from tsplib_parser import City

class TripEval:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.eval_count = 0
        self.best_length = float('inf')
        self.history = []
        self.max_evals = 30_000


    def evaluate(self, trip: list[int]):
        if self.is_budget_exhausted():
            return
        self.eval_count += 1

        current_trip_length = 0
        for city_index in range(len(trip) - 1):
            distance = self.distance_matrix[trip[city_index]][trip[city_index+1]]
            current_trip_length += distance

        # add return trip distance
        current_trip_length += self.distance_matrix[trip[-1]][trip[0]]

        if current_trip_length < self.best_length:
            self.best_length = current_trip_length
            self.best_trip = list(trip)

        self.history.append([self.eval_count, self.best_length])
        return current_trip_length

    def is_budget_exhausted(self) -> bool:
        if self.eval_count >= self.max_evals:
            return True
        return False

def calc_distance(city_a: City, city_b: City) -> int:
    return floor(sqrt((city_a.x - city_b.x) ** 2 + (city_a.y - city_b.y) ** 2) + 0.5)

def create_distance_matrix(cities: list[City]) -> list[list[int]]:
    n = len(cities)
    distance_matrix = [[None for _ in range(n)] for _ in range(n)]
    for i in range(0,n):
        for j in range(0,n):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = calc_distance(cities[i], cities[j])
    return distance_matrix

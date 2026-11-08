from typing import Callable

from operators import generate_initial_trip, swap_operator, reverse_operator
from distance import TripEval


class Bee:
    def __init__(self, trip: list[int], length: float):
        self.trip = trip
        self.length = length
        
class BeesAlgorithm:
    def __init__(
        self,
        num_cities: int,
        evaluator: TripEval,
        operator: Callable[[list[int]], list[int]],
        population_size: int = 30,    # n: Total scout bees
        selected_sites: int = 10,     # m: Number of best sites selected
        elite_sites: int = 3,         # e: Number of elite sites (subset of m)
        recruited_elite: int = 10,    # ne: Foragers recruited for each elite site
        recruited_selected: int = 5   # ns: Foragers recruited for remaining (m - e) sites
    ):
        self.num_cities = num_cities
        self.evaluator = evaluator
        self.operator = operator
        self.population_size = population_size
        self.selected_sites = selected_sites
        self.elite_sites = elite_sites
        self.recruited_elite = recruited_elite
        self.recruited_selected = recruited_selected

    def init_population(self) -> list[Bee]:
        population = []

        for _ in range(0,self.population_size):
            trip = generate_initial_trip(self.num_cities)
            trip_length = self.evaluator.evaluate(trip)
            bee = Bee(trip, trip_length)
            population.append(bee)

        return population


    def search_patch(self, site: Bee, num_foragers: int) -> Bee:
        best_bee_in_patch = site

        for _ in range(0,num_foragers):
            if self.evaluator.is_budget_exhausted():
                break
            mutated_trip = self.operator(site.trip.copy())
            mutated_trip_length = self.evaluator.evaluate(mutated_trip)
            if mutated_trip_length < best_bee_in_patch.length:
                best_bee_in_patch = Bee(mutated_trip.copy(),mutated_trip_length)

        return best_bee_in_patch


    def run(self) -> tuple[list[int], int, list]:
        population = self.init_population()

        while not self.evaluator.is_budget_exhausted():
            population.sort(key=lambda b: b.length)
            new_population = []
            elite_trips = population[:self.elite_sites]
            for bee in elite_trips:
                new_population.append(self.search_patch(bee, self.recruited_elite))
            next_best_trips = population[self.elite_sites:self.selected_sites]
            for bee in next_best_trips:
                new_population.append(self.search_patch(bee, self.recruited_selected))
            for _ in range(0,(self.population_size- len(new_population))):
                trip = generate_initial_trip(self.num_cities)
                trip_length = self.evaluator.evaluate(trip)
                bee = Bee(trip, trip_length)
                new_population.append(bee)
            population = new_population

        return self.evaluator.best_trip, self.evaluator.best_length, self.evaluator.history


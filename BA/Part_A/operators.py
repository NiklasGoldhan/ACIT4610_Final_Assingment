import random


def generate_initial_trip(num_cities: int) -> list[int]:
    possible_trip = []
    for i in range(1,num_cities):
        possible_trip.append(i)

    random.shuffle(possible_trip)

    # add city 1 (index 0) as start
    possible_trip.insert(0,0)
    return possible_trip


def swap_operator(trip: list[int]) -> list[int]:
    random_index_1, random_index_2 = random.sample(range(1, len(trip)), k=2)

    new_trip = trip.copy()
    new_trip[random_index_1] = trip[random_index_2]
    new_trip[random_index_2] = trip[random_index_1]

    return new_trip


def reverse_operator(trip: list[int]) -> list[int]:
    idx_a, idx_b = random.sample(range(1, len(trip)), k=2)
    random_index_1, random_index_2 = min(idx_a, idx_b), max(idx_a, idx_b)

    new_trip = trip.copy()
    part_to_reverse = new_trip[random_index_1:random_index_2]
    part_to_reverse.reverse()
    new_trip[random_index_1:random_index_2] = part_to_reverse

    return new_trip

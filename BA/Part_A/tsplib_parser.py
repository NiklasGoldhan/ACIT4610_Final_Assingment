class City:
    id: int = None
    x: float = None
    y: float = None

    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y

class TSPInstance:
    def __init__(self):
        self.metadata: dict = {}
        self.dataset: list = []

    def read_file(self,path):
        with open(path, 'r', encoding='utf-8') as file:
            in_node_section = False
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line == 'EOF':
                    break
                if line == 'NODE_COORD_SECTION':
                    in_node_section = True
                    continue
                if not in_node_section:
                    # Parse metadata header
                    if ":" in line:
                        key, value = line.split(":", 1)
                        self.metadata[key.strip()] = value.strip()
                else:
                    # Parse coordinate rows
                    parts = line.split()
                    city = City(int(parts[0])-1, float(parts[1]), float(parts[2]))
                    self.dataset.append(city)

    def validate_dataset_metadata(self) -> bool:
        if self.metadata.get('NAME') == None:
            return False
        if self.metadata.get('TYPE') != 'TSP':
            return False 
        if self.metadata.get('EDGE_WEIGHT_TYPE') != 'EUC_2D': 
            return False
        if int(self.metadata.get('DIMENSION')) != len(self.dataset):
            return False
        city_index_set = None
        city_index_list = []
        for city in self.dataset:
            city_index_list.append(city.id)
        city_index_set = set(city_index_list)
        if len(city_index_set) != len(self.dataset):
            return False
        return True

tspinstance = TSPInstance()

tspinstance.read_file('BA/Part_A/data/eil51.tsp')
tspinstance.validate_dataset_metadata()
            
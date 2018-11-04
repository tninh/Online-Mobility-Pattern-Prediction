import csv
from markov import Markov

map_trajectory = {}
ap2id_map = {}
filter_len = 10

class Position:
    ts = ''
    ap = ''
    map_location = ''

    def __init__(self, ts, ap, ap_id, map_location):
        self.ts = ts
        self.ap = ap
        self.ap_id = ap_id
        self.map_location = map_location
    def display(self):
        return self.ts + ' ' + self.ap + ' ' + str(self.ap_id) + ' ' + self.map_location

def build_aggregate_map_from_csv():
    global map_trajectory
    global ap2id_map

    with open('ap.csv', newline='') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            uname = row[0]
            ts = row[3]
            ap = row[5]
            map_location = row[8]
            ap_id = 0
            if ap in ap2id_map.keys():
                ap_id = ap2id_map[ap]
            else:
                ap_id = count
                ap2id_map[ap] = count
                count += 1
            if uname in map_trajectory.keys():
                map_trajectory[uname].append(Position(ts, ap, ap_id, map_location))
            else:
                map_trajectory[uname] = [Position(ts, ap, ap_id, map_location),]

def filter_short_records():
    global map_trajectory
    filtered_map = {}
    for key, value in map_trajectory.items():
        if len(value) >= filter_len :
            filtered_map[key] = value
    map_trajectory = filtered_map

def sort_map_by_ts():
    global map_trajectory
    for l in map_trajectory.values():
        l.sort(key=lambda x: x.ts)

def print_map():
    for key, value in map_trajectory.items():
        print(key)
        print([x.ap_id for x in value])

def generate_test_data():
    new_map = {}
    for key, value in map_trajectory.items():
        new_map[key] = [x.ap_id for x in value]
    return new_map

def predict(test_data):
    markov = Markov(his_count=1, pre_count=3, test_count=3, tol=3)
    markov.read(test_data)
    return markov.get_predict_map(test_data)

def main():
    build_aggregate_map_from_csv()
    filter_short_records()
    sort_map_by_ts()
    test_data = generate_test_data()
    print_map()
    print("------------------------- start using markov model --------------------------- ")
    predict_map = predict(test_data)
    for key, value in predict_map.items():
        print(key)
        print(value)

if __name__ == "__main__":
    main()

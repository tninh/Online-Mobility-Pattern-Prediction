def count_path_occurrence(raw_string_list, substring_list):
    count = 0
    for i in range(len(raw_string_list)):
        for j in range(len(raw_string_list[i])):
            if raw_string_list[i][j:j + len(substring_list)] == substring_list:
                count += 1
    print(count)

#figure out how many times substring occurs in raw string,
# exactly matching (time -> ap) pair
def count_path_occurrence_with_time(raw_string_dict, substring_dict):
    count = 0
    for day_dict in raw_string_dict.values():
        if substring_dict.items() <= day_dict.items():
            count += 1
    print(count)


raw = {'day1': {'5:00': '80', '6:00': '78', '7:00': '99', '8:00': '209'},
       'day2': {'5:00': '80', '6:00': '78', '7:00': '99', '8:00': '22'},
       'day3': {'4:00': '80', '5:00': '78', '6:00': '99'},
       'day4': {'4:00': '6', '5:00': '80', '6:00': '78', '7:00': '100'},
       'day5': {'5:00': '80', '6:00': '80', '7:00': '99'}}

sub = {'4:00': '80', '5:00': '78', '6:00': '99'}
count_path_occurrence_with_time(raw, sub)
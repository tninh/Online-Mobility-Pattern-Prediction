def count_path_occurrence(raw_string_list, substring_list):
    count = 0
    for i in range(len(raw_string_list)):
        for j in range(len(raw_string_list[i])):
            if raw_string_list[i][j:j + len(substring_list)] == substring_list:
                count += 1
    print(count)


# def count_path_occurrence_with_time():


a = [['a', '1', 'c'], ['9', 'a', '1', 'd']]
b = ['a', '1']
count_path_occurrence(a, b)

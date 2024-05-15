from data.data_tools import read_h5_file
data = read_h5_file('dataset/new.h5')
# max_len = 0
for vi in data.keys():
    print(data[vi]['gtsummary'])
    # print(data[vi]['gtsummary'].shape)
    # print(data[vi]['change_points'].shape)
    # print(data[vi]['gtscore'].shape)
    break

        
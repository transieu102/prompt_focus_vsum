from data.data_tools import read_h5_file
data = read_h5_file('dataset/eccv16_dataset_tvsum_google_pool5_sumprompt_clip_L14.h5')
for vi in data.keys():
    for key in data[vi].keys():
        print(key)

    break
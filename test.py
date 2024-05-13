from data.data_tools import read_h5_file
data = read_h5_file('dataset/eccv16_dataset_tvsum_google_pool5_sumprompt_clip_L14.h5')
max_len = 0
for vi in data.keys():
    lenz = len(data[vi]['picks'][()])
    if lenz > max_len:
        max_len = lenz
print(max_len)
        
import pickle
import os

root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/'
_f = open(os.path.join(root, 'pickle/test2_lmark2img.pkl'), 'rb')
pickle_data = pickle.load(_f)
_f.close()

print('original:{}'.format(len(pickle_data)))

new_pickle_data = []
for data in pickle_data:
    ani_path = os.path.join(root, 'test', data[0], data[1][:5]+'_ani.mp4')
    if os.path.exists(ani_path):
        new_pickle_data.append(data)

print('new:{}'.format(len(new_pickle_data)))
_f = open(os.path.join(root, 'pickle/test3_lmark2img.pkl'), 'wb')
pickle.dump(new_pickle_data, _f)
_f.close()
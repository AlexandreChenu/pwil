import pickle
import numpy as np


L_demo = []
for i in range(1,7):

    with open('./demonstrations/' + str(i) + '.demo', 'rb') as handle:
        demo = pickle.load(handle)

    new_demo = []
    for obs, action in zip(demo["obs"],demo["actions"]):
        new_demo.append({"observation":obs, "action":action})

    L_demo.append(new_demo)

print(len(L_demo))


with open('./demonstrations/fetchenv.pkl', 'wb') as handle:
    pickle.dump(L_demo, handle, protocol=pickle.HIGHEST_PROTOCOL)

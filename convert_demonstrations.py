import pickle5 as pickle
import numpy as np


L_demo = []
for i in range(1,3):

    with open('./demonstrations/dubins/' + str(i) + '.demo', 'rb') as handle:
        demo = pickle.load(handle)

    print("demo = ", demo)
    new_demo = []
    for obs, action in zip(demo["obs"],demo["actions"]):
        print("obs = ", obs)
        transition = {"observation":np.array(obs).astype(np.float32), "action":action.astype(np.float32), "reward":np.array(0.,dtype=np.float32)}
        print("transition = ", transition)
        new_demo.append(transition)


    L_demo.append(new_demo)

print(len(L_demo))


with open('./demonstrations/DubinsMazeEnv-v0.pkl', 'wb') as handle:
    pickle.dump(L_demo, handle, protocol=4)

import pickle5 as pickle
import numpy as np


L_demo = []
for i in range(1,2):

	with open('./demonstrations/humanoidenv/' + str(i) + '.demo', 'rb') as handle:
		demo = pickle.load(handle)

	print("demo.keys() = ", demo.keys())
	new_demo = []
	for obs, action in zip(demo["observations"],demo["actions"]):
		print("obs shape = ", np.array(obs).shape)
		print("action shape = ", np.array(action).reshape(-1,).shape)

		transition = {"observation":np.array(obs).astype(np.float32), "action":action.reshape(-1,).astype(np.float32), "reward":np.array(0.,dtype=np.float32)}
		# print("transition = ", transition)
		new_demo.append(transition)


	L_demo.append(new_demo)

print(len(L_demo))


with open('./demonstrations/humanoidenv/HumanoidEnv-v0.pkl', 'wb') as handle:
	pickle.dump(L_demo, handle, protocol=4)

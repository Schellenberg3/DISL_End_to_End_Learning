import pickle
import numpy as np
import os

root = 'networks/imitation'
dirs = os.listdir(root)
head = 'steps, mse'
print(dirs)
for d in dirs:
    try:
        with open(f'{root}/{d}/train_performance.plk', 'rb') as fp:
            h = pickle.load(fp)

        steps = []
        mse = []
        for i in h:
            steps.append(i['steps'])
            mse.append(i['mse'][0])

        a = np.vstack((np.asarray(steps), np.asarray(mse))).transpose()

        np.savetxt(f'{root}/{d}/train_performance.csv', a, delimiter=",", header=head)
        print(f'Pkl renamed in {d}')

    except:
        print(f'Not found: {root}/{d}/train_performance.plk')
        pass


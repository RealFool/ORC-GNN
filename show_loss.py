import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

losses = np.load('boundary_loss.npy')

delta = np.load('delta_list.npy')

# plt.plot(ce_loss)
# plt.title('Loss Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()

for i in range(delta.shape[1]):
    plt.plot(delta[:, i], label=f'Delta {i+1}')

plt.title('Delta Curves')
plt.xlabel('Time or Epochs')
plt.ylabel('Delta Value')
plt.legend()
plt.show()
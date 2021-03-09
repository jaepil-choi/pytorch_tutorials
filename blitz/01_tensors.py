# https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

# Tensors
#%%
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#%%
## Tensor Initialization

data_l = [
    [1, 2],
    [3, 4]
]

data_arr = np.array(data_l)
data1_tensor = torch.from_numpy(data_arr)

data2_tensor = torch.ones_like(data1_tensor)

#%%
shape = (2, 3)
rand_tensor = torch.rand(shape)
# %%
## Tensor Attributes

print(f'''
shape: {rand_tensor.shape}
dtype: {rand_tensor.dtype}
stored: {rand_tensor.device}
''')
# %%
## Tensor Operations

rand = rand_tensor.to(device)
rand
# %%
ones = torch.ones((4, 4))
ones[:, 1] = 0
ones

# %%
torch.cat([ones, ones, ones], dim=1)
# %%
rand * rand
# %%
rand @ rand.T
# %%
### in-place operations have a _ suffix. 

print(rand)
rand.add_(5)
print(rand)

# %%
## Bridge with Numpy
print(ones)
ones_arr = ones.numpy()
ones_arr
# %%
ones.add_(99)
print(ones_arr) # Change in tensor reflects to numpy and vice versa. 
# %%

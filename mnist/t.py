import numpy as np
a=np.zeros((100,28,32))
b=np.zeros((100,28,32))

k=np.einsum("btv,bvs->bts",a,b.transpose(0,2,1))
print(k.shape)
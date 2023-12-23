import torch

seqlen = 10
n_heads = 32
n_kv_heads = 8
n_dim = 128
Q = torch.randn(seqlen,n_heads,n_dim)
K = torch.randn(seqlen,n_kv_heads,n_dim)

attn = torch.matmul(Q, K.transpose(1,2))

ql = Q.tolist()
qk = K.tolist()

out = torch.empty(seqlen,n_heads,n_kv_heads)
once = 0
for i in range(seqlen):
  for j in range(n_heads):
    for k in range(n_kv_heads):
      if once == 0:
        print(Q[i].shape)
        print(K[i].shape)
      once += 1
      out[i,j,k] = (Q[i][j] * K[i][k]).sum()

assert int((out-attn).sum())==0


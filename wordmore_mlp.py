#!/usr/bin/env python
# coding: utf-8

# In[362]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


torch.cuda.is_available()


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[29]:


words = open("data_files/words.txt").read().splitlines()


# In[30]:


chars = sorted(list(set("".join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


# In[284]:


dim = 10
block_size = 3
emb_shape = block_size * dim
emb_shape


# In[285]:


def build_dataset(block_size=3) -> torch.tensor:
    # block_size = 3  # Context length
    X, Y = [], []
    for word in words:
        context = [0]*block_size
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1: ] + [ix]

    X = np.array(X)
    Y = np.array(Y)

    xtrn, xtst_, ytrn, ytst_ = train_test_split(X, Y, test_size=0.2, random_state=1111)
    xval, xtst, yval, ytst = train_test_split(xtst_, ytst_, test_size=0.5, random_state=1111)
    
    xtrn = torch.tensor(xtrn, device=device)
    ytrn = torch.tensor(ytrn, dtype=torch.long, device=device)
    xval = torch.tensor(xval, device=device)
    yval = torch.tensor(yval, dtype=torch.long, device=device)
    xtst = torch.tensor(xtst, device=device)
    ytst = torch.tensor(ytst, dtype=torch.long, device=device)
    
    return xtrn, ytrn, xval, yval, xtst, ytst

xtrn, ytrn, xval, yval, xtst, ytst = build_dataset(block_size)


# In[288]:


# Initializing embedding space and parameters

torch.manual_seed(1111)
C = torch.randn((27, dim),device=device)  # 2 is the dimension of embedding space
w1 = torch.randn((emb_shape, 200),device=device)
b1 = torch.randn(200, device=device)
w2 = torch.randn((200, 200), device=device)
b2 = torch.randn(200, device=device)
w3 = torch.randn((200, 27), device=device)
b3 = torch.randn(27, device=device)

parameters = [C, w1, b1, w2, b2, w3, b3]


# In[289]:


sum(p.nelement() for p in parameters)


# In[290]:


for p in parameters:
    p.requires_grad = True


# In[291]:


loss_fn = nn.CrossEntropyLoss()


# In[353]:


epoch_list = []
loss_list = []
epochs = 20000
# epochs = 1000
for epoch in range(epochs):

    ix = torch.randint(0, xtrn.shape[0], (32,))
    
    # Forward Pass
    emb = C[xtrn[ix]]
    h1 = torch.tanh(emb.view(-1, emb_shape) @ w1 + b1)
    h2 = torch.tanh(h1 @ w2 + b2)
    logits = h2 @ w3 + b3
    
    # Backprop
    for p in parameters:
        p.grad = None
    loss = loss_fn(logits, ytrn[ix])
    loss.backward()
    
    # Gradient Descent
    # lr = 0.1 if epoch < 1000 else 0.05
    lr = 0.0001
    for p in parameters:
        p.data += -lr * p.grad

    # Tracking stats
    epoch_list.append(epoch)
    loss_list.append(loss.log10().item())

    # Printing stats
    
    if epoch%2000 == 0:
        print(f"epoch: {epoch} | loss: {loss} | lr: {lr}")
print(f"epoch: {epoch+1} | loss: {loss} | lr: {lr}")


# In[354]:


plt.figure(figsize=(3, 3))
plt.plot(epoch_list, loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")


# In[355]:


emb = C[xval]
h1 = torch.tanh(emb.view(-1, emb_shape) @ w1 + b1)
h2 = torch.tanh(h1 @ w2 + b2)
logits_val = h2 @ w3 + b3
loss_val = loss_fn(logits_val, yval)
print(f"Validation Set loss: {loss_val}")


# In[356]:


emb = C[xtst]
h1 = torch.tanh(emb.view(-1, emb_shape) @ w1 + b1)
h2 = torch.tanh(h1 @ w2 + b2)
logits_tst = h2 @ w3 + b3
loss_tst = loss_fn(logits_tst, ytst)
print(f"Test Set loss: {loss_tst}")


# In[382]:


# g = torch.Generator().manual_seed(2147483647 + 10)
# torch.manual_seed(1111)
for _ in range(20):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h1 = torch.tanh(emb.view(1, -1) @ w1 + b1)
      h2 = torch.tanh(h1 @ w2 + b2)
      logits = h2 @ w3 + b3
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))


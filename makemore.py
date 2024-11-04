import torch
import matplotlib.pyplot as plt


def plot(N, itos):
#    %matplotlib inline
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')


    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

    plt.axis('off')
    plt.tight_layout()
    plt.show()



words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))

stoi = {s: i+1 for i, s in enumerate(chars) }
stoi['.'] = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

itos = {i : s for s, i in stoi.items()}
#plot(N, itos)
# g = torch.Generator().manual_seed(2147483647)

P = N.float()
P = P / P.sum(1, keepdim=True)


for _ in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = int(torch.multinomial(p, num_samples=1, replacement=True).item())
        out.append(itos[ix])

        if ix == 0: 
            break

    print(''.join(out))


# calculate the likelyhood

log_lh = 0.0
n = 0

for w in words:
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_lh -= torch.log(prob)
        n += 1

print(f"{log_lh=}")
print(f"{log_lh/n=}")




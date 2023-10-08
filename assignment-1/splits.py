def splits(ops, amt):
    out = []
    a = ops // amt
    b = ops % amt
    nxt = 0
    for i in range(amt):
        start = nxt
        nxt += a
        if i < b:
            nxt += 1
        out.append((start, nxt))
    return out

def get_split(ops, amt, i):
    a = ops // amt
    b = ops % amt
    start = a*i + min(i,b)
    end = start + a + (i<b)
    return start, end

print(splits(18, 4))
for i in range(4):
    print(get_split(18, 4, i))

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	b = []
    for i in range(len(a[0])):
        l = []
        for j in range(len(a)):
            l.append(a[j][i])
        b.append(l)
    return b
def knapsack(items, maxweight):
    N = len(items)
    W = maxweight

    bestvalues = [[0] * (W + 1)
                  for i in range(N + 1)]

    for i, (value, weight) in enumerate(items):

        for capacity in range(maxweight + 1):

            if weight > capacity:
                bestvalues[i + 1][capacity] = bestvalues[i][capacity]
            else:
                candidate1 = bestvalues[i][capacity]
                candidate2 = bestvalues[i][capacity - weight] + value
                bestvalues[i + 1][capacity] = max(candidate1, candidate2)

    reconstruction = []
    j = maxweight
    for i in range(N, 0, -1):
        if bestvalues[i][j] != bestvalues[i - 1][j]:
            reconstruction.append(i - 1)
            j -= items[i - 1][1]

    reconstruction.reverse()

    return bestvalues[len(items)][maxweight], reconstruction
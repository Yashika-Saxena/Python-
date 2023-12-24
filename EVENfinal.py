def calculate_score(num_test_cases, marble_counts, ram_marbles, karan_marbles):
    scores = []

    for i in range(num_test_cases):
        ram_score = 0
        karan_score = 0

        for j in range(marble_counts[i]):
            
            if j % 2 == 0:
                ram_score += ram_marbles[i][j]
                karan_marbles[i][j] = 0
            else:
                karan_score += karan_marbles[i][j]
                ram_marbles[i][j] = 0
        scores.append(ram_score - karan_score)

    return scores
test_cases = int(input("Enter the number of test cases: "))

marble_counts = []
ram_marbles = []
karan_marbles = []

for i in range(test_cases):
    marble_counts.append(int(input(f"Enter the number of marbles for {i+1} test case: ")))
    ram_marbles.append(list(map(int, input(f"Enter Ram's marbles for {i+1} test case: ").split())))
    karan_marbles.append(list(map(int, input(f"Enter Karan's marbles for {i+1} test case: ").split())))

scores = calculate_score(test_cases, marble_counts, ram_marbles, karan_marbles)

for i, score in enumerate(scores, start=1):
    print(f"Score for test case {i}: {score}")

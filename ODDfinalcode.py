# import sys
def calculate_score(ram_marbles, karan_marbles):
    return sum(ram_marbles) - sum(karan_marbles)

def play_marbles_game(T, N, ram_marbles, karan_marbles):
    scores = []

    for i in range(T):
        current_marbles = [ram_marbles[i], karan_marbles[i]]
        current_turn = 0  # 0 for Ram's turn, 1 for Karan's turn

        while True:
            player_marbles = current_marbles[current_turn]
            opponent_marbles = current_marbles[1 - current_turn]

            if sum(player_marbles) == 0:
                break 

            # lowest marble for the current playert
            player_lowest_index = player_marbles.index(min(player_marbles))
            opponent_highest_index = opponent_marbles.index(max(opponent_marbles))

            # Reduce marbles
            player_marbles[player_lowest_index] -= 1
            opponent_marbles[opponent_highest_index] = 0
            if current_turn ==0:
                print("Ram turn :]")
            else:
                print("karan turn")
                
            print(ram_marbles)
            print(karan_marbles)
            print("=========================================")
            current_turn = 1 - current_turn  # Switch turns
            
        scores.append(calculate_score(ram_marbles[i], karan_marbles[i]))

    return scores
test_cases = int(input("Enter the number of test cases: "))

ram_marbles_list = []
karan_marbles_list = []

for i in range(test_cases):
    marble_colors = int(input(f"Enter the number of marble colors for test case {i+1}:  "))
    ram_marbles = list(map(int, input(f"Enter RAM's marbles for test case {i+1}: ").split()))
    karan_marbles = list(map(int, input(f"Enter KARAN's marbles for test case {i+1}: ").split()))
    
    ram_marbles_list.append(ram_marbles)
    karan_marbles_list.append(karan_marbles)

# Calculate scores and print result
scores = play_marbles_game(test_cases,marble_colors, ram_marbles_list, karan_marbles_list)
print(' '.join(map(str, scores)))

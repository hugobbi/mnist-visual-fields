act_list = [(5, 1.2), (3, 1.3), (2, 1.4), (1, 1.5), (4, 1.6)]

k = 3

string = ''
for i in range(k):
    eol = '\n' if i != k - 1 else ''
    string += f'{act_list[i][0]}: {act_list[i][1]:.4f}{eol}'

print(string)
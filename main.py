def do_try(i):
    if (vx_arr[i] == 1):
        return False
    vx_arr[i] = True
    for j in range(0,n):
        if (matrix_ef[i][j] - max_row[i] - min_col[j]) == 0:
            vy_arr[j] = True
    for j in range(0,n):
        if ((matrix_ef[i][j] - max_row[i] - min_col[j]) == 0) and yx_arr[j] == -1:
            xy_arr[i] = j
            yx_arr[j] = i
            return True
    for j in range(0,n):
        if ((matrix_ef[i][j] - max_row[i] - min_col[j]) == 0) and do_try(yx_arr[j]):
            xy_arr[i] = j
            yx_arr[j] = i
            return True
    return False

n = 5

matrix_ef = [[7,3,6,9,5],[7,5,7,5,6],[7,6,8,8,9],[3,1,6,5,7],[2,4,9,9,5]]

xy_arr = []
yx_arr = []

vx_arr = []
vy_arr = []

max_row = [0]*n
min_row = [0]*n
min_col = [0]*n

for i in range(0, n):
    for j in range(0, n):
        max_row[i] = max(max_row[i], matrix_ef[i][j])


xy_arr = [-1]*n
yx_arr = [-1]*n
c = 0

while c < n:
    vx_arr = [0]*n
    vy_arr = [0]*n
    k = 0
    for i in range(0,n):
        if xy_arr[i] == -1 and do_try(i):
            k += 1
    c += k
    if k == 0:
        z = 9999999999999
        for i in range(0,n):
            if (vx_arr[i]):
                for j in range(0,n):
                    if (not(vy_arr[j])):
                        z = min(z, max_row[i] + min_col[j] - matrix_ef[i][j])
        for i in range(0,n):
            if (vx_arr[i]):
                max_row[i] -= z
            if vy_arr[i]:
                min_col[i] += z

res = 0
for i in range(0,n):
    res += matrix_ef[i][xy_arr[i]]
print(res)
for i in range(0,n):
    print (xy_arr[i] + 1)

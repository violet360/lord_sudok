sudoku =[[3, 0, 6, 5, 0, 8, 4, 0, 0], 
         [5, 2, 0, 0, 0, 0, 0, 0, 0], 
         [0, 8, 7, 0, 0, 0, 0, 3, 1], 
         [0, 0, 3, 0, 1, 0, 0, 8, 0], 
         [9, 0, 0, 8, 6, 3, 0, 0, 5], 
         [0, 5, 0, 0, 9, 0, 6, 0, 0], 
         [1, 3, 0, 0, 0, 0, 2, 5, 0], 
         [0, 0, 0, 0, 0, 0, 0, 7, 4], 
         [0, 0, 5, 2, 0, 6, 3, 0, 0]] 



def solve(sudoku):
	var = init(sudoku)
	if var is None:
		return True
	else:
		row, col = var

	for i in range(1,10):
		if check(sudoku, i, (row, col)):
			sudoku[row][col] = i

			if solve(sudoku):
				return True

			sudoku[row][col] = 0

	return False


def check(bo, num, pos):
    # Check row
	for i in range(len(bo[0])):
		if bo[pos[0]][i] == num and pos[1] != i:
			return False

    # Check column
	for i in range(len(bo)):
		if bo[i][pos[1]] == num and pos[0] != i:
 			return False

    # Check box
	box_x = pos[1] // 3
	box_y = pos[0] // 3

	for i in range(box_y*3, box_y*3 + 3):
		for j in range(box_x * 3, box_x*3 + 3):
			if bo[i][j] == num and (i,j) != pos:
				return False

	return True


def init(sudoku):
	var  = None
	for i in range(len(sudoku)):
		for j in range(len(sudoku[0])):
			if sudoku[i][j] == 0:
				var = i, j
				return var

print(sudoku)
solve(sudoku)
print(sudoku)
from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()

    # ====> insert your code below here
    # Loop through all possible values for each tumbler
    for tumbler1 in puzzle.value_set:  # First tumbler
        for tumbler2 in puzzle.value_set:  # Second tumbler
            for tumbler3 in puzzle.value_set:  # Third tumbler
                for tumbler4 in puzzle.value_set:  # Fourth tumbler
                    # Assign the current combination to the candidate
                    my_attempt.variable_values = [tumbler1, tumbler2, tumbler3, tumbler4]

                    # Attempt to evaluate the combination
                    try:
                        score = puzzle.evaluate(my_attempt.variable_values)
                        # If score is 1, the combination is correct
                        if score == 1:
                            return my_attempt.variable_values
                    except ValueError as error:
                        # Log any invalid combination errors
                        print(f"Error in combination: {error}")


    # <==== insert your code above here

    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
    # Loop through each row in the array
    for index in range(namearray.shape[0]):
        # Extract the last 6 columns of the row
        name_slice = namearray[index, -6:]
        # Convert the slice to a single string
        name_string = "".join(name_slice)
        # Add the string to the result list
        family_names.append(name_string)



    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays

    # ====> insert your code below here

    # Ensure the array is 9x9
    assert attempt.shape == (9, 9), "Input must be a 9x9 array"

    # Collect all 9 row slices
    for row_idx in range(9):
        slices.append(attempt[row_idx, :])

    # Collect all 9 column slices
    for col_idx in range(9):
        slices.append(attempt[:, col_idx])

    for row_start in range(0, 9, 3):
        for col_start in range(0, 9, 3):
            sub_grid = attempt[row_start:row_start+3, col_start:col_start+3]
            slices.append(sub_grid)

    # Evaluate each slice for uniqueness
    for grid_slice in slices:
        distinct_values = np.unique(grid_slice)
        if len(distinct_values) == 9:
            tests_passed += 1

    # <==== insert your code above here
    # return count of tests passed
    return tests_passed

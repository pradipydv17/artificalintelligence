
from approvedimports import *

class DepthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "depth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """void in superclass
        In sub-classes should implement different algorithms
        depending on what item it picks from self.open_list
        and what it then does to the openlist

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        # if list is empty, return next_soln
        if len(self.open_list) == 0:
            return next_soln
        reversed_list = self.open_list[::-1]
        next_soln = reversed_list[0]
        # update list without that item
        self.open_list = reversed_list[1:][::-1]


        # <==== insert your pseudo-code and code above here
        return next_soln

class BreadthFirstSearch(SingleMemberSearch):
    """your implementation of depth first search to extend
    the superclass SingleMemberSearch search.
    Adds  a __str__method
    Over-rides the method select_and_move_from_openlist
    to implement the algorithm
    """

    def __str__(self):
        return "breadth-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements the breadth-first search algorithm

        Returns
        -------
        next working candidate (solution) taken from openlist
        """
        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        # return None if list is empty
        if len(self.open_list) == 0:
            return None
        # take first item for level-by-level search
        next_soln = self.open_list[0]
        # keep rest of list
        self.open_list = self.open_list[1:]

        # <==== insert your pseudo-code and code above here
        return next_soln

class BestFirstSearch(SingleMemberSearch):
    """Implementation of Best-First search."""

    def __str__(self):
        return "best-first"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements Best First by finding, popping and returning member from openlist
        with best quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        # if list is empty, return empty solution
        if len(self.open_list) == 0:
            return next_soln
        # pick first item as best to start
        next_soln = self.open_list[0]
        # check each item to find better quality
        for sol in self.open_list:
            if sol.quality < next_soln.quality:
                next_soln = sol
        # make new list without the picked item
        new_list = []
        for sol in self.open_list:
            if sol != next_soln:
                new_list.append(sol)
        self.open_list = new_list
        # <==== insert your pseudo-code and code above here
        return next_soln

class AStarSearch(SingleMemberSearch):
    """Implementation of A-Star  search."""

    def __str__(self):
        return "A Star"

    def select_and_move_from_openlist(self) -> CandidateSolution:
        """Implements A-Star by finding, popping and returning member from openlist
        with lowest combined length+quality.

        Returns
        -------
        next working candidate (solution) taken from openlist
        """

        # create a candidate solution variable to hold the next solution
        next_soln = CandidateSolution()

        # ====> insert your pseudo-code and code below here
        # if list has no items, return empty solution
        if len(self.open_list) == 0:
            return next_soln
        # start with first item as best
        next_soln = self.open_list[0]
        best_score = len(next_soln.variable_values) + next_soln.quality
        # look at each item to find better score
        for i in range(len(self.open_list)):
            score = len(self.open_list[i].variable_values) + self.open_list[i].quality
            if score < best_score:
                next_soln = self.open_list[i]
                best_score = score
        # build new list without picked item
        new_list = []
        for sol in self.open_list:
            if sol != next_soln:
                new_list.append(sol)
        self.open_list = new_list

        # <==== insert your pseudo-code and code above here
        return next_soln
wall_colour= 0.0
hole_colour = 1.0

def create_maze_breaks_depthfirst():
    # ====> insert your code below here
    #remember to comment out any mention of show_maze() before you submit your work
    # Load base maze
    maze = Maze(mazefile="maze.txt")

    maze.contents[3][4] = hole_colour  # Open path to trick DFS
    maze.contents[8][4] = wall_colour  # Block DFS at the end

    maze.contents[10][6] = hole_colour  # Another DFS trap
    maze.contents[14][6] = wall_colour  # Dead-end
    maze.contents[16][1] = hole_colour  # Dead-end
    maze.contents[19][4] = hole_colour  # Dead-end

    maze.contents[8][1] = hole_colour
    maze.contents[12][9] = wall_colour
    maze.contents[11][12] = wall_colour
    maze.contents[9][2] = wall_colour
    maze.contents[10][19] = wall_colour
    maze.contents[18][5] = wall_colour




    # Save the maze
    maze.save_to_txt("maze-breaks-depth.txt")

    # <==== insert your code above here

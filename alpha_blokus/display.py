import numpy as np
from matplotlib import pyplot as plt

class Display:
    def __init__(self, occupancies, overlay_dots=None, board_size: int=20):
        self.occupancies = occupancies
        self.overlay_dots = overlay_dots
        self.board_size = board_size

    def show(self):
        grid = np.zeros((self.board_size, self.board_size, 3))
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.occupancies[0, x, y]:
                    color = [0, 0, 1]
                elif self.occupancies[1, x, y]:
                    color = [1, 1, 0]
                elif self.occupancies[2, x, y]:
                    color = [1, 0, 0]
                elif self.occupancies[3, x, y]:
                    color = [0, 1, 0]
                else:
                    color = [1, 1, 1]
                grid[x, y] = color

        # Plot the grid
        plt.imshow(grid, interpolation='nearest')
        plt.axis('on')  # Show the axes
        plt.grid(color='black', linestyle='-', linewidth=2)  # Add gridlines

        if self.overlay_dots is not None:
            x_coords, y_coords = [], []
            for x in range(self.board_size):
                for y in range(self.board_size):
                    if self.overlay_dots[x, y]:
                        x_coords.append(x)
                        y_coords.append(y)

            plt.scatter(y_coords, x_coords, color='black', s=20)  # Draw black dots

        # Adjust the gridlines to match the cells
        plt.xticks(np.arange(-0.5, self.board_size, 1), [])
        plt.yticks(np.arange(-0.5, self.board_size, 1), [])
        plt.gca().set_xticks(np.arange(-0.5, self.board_size, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, self.board_size, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=1)

        plt.show()
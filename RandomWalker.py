import numpy as np
import matplotlib.pyplot as plt

class RandomWalker:
    def __init__(self, start_cords, speed, max_x, max_y):
        # x = cords[0], y = cords[1]
        start_cords = list(start_cords)
        self._current_cords = start_cords[:]
        self._all_cords = [start_cords[:]]

        self._speed = speed
        self._max_x = max_x
        self._max_y = max_y

    def step(self):
        # Sampling direction uniformly
        direction = np.random.uniform(0, 2 * np.pi)

        self._current_cords[0] += np.cos(direction) * self._speed
        self._current_cords[1] += np.sin(direction) * self._speed

        # Setting the boundaries
        self._current_cords[0] = np.maximum(self._current_cords[0], 0)
        self._current_cords[0] = np.minimum(self._current_cords[0], self._max_x)

        # Setting the boundaries
        self._current_cords[1] = np.maximum(self._current_cords[1], 0)
        self._current_cords[1] = np.minimum(self._current_cords[1], self._max_y)

        self._all_cords.append(list(self._current_cords))

    def current_cords(self):
        return np.reshape(self._current_cords, (-1, 2))


def main():
    rw = RandomWalker((500, 500), 1, 1000, 1000)

    for i in range(5000):
        rw.step()

    all_cords = np.array(rw._all_cords)


    plt.plot(all_cords[:, 0], all_cords[:, 1])
    plt.show()


if __name__ == "__main__":
    main()

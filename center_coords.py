import numpy as np

def center_coords(ship_position, ship_heading, asteroid_positions, map_size):
    """
    Given a ship's position and heading, find the polar coordinates of all asteroids relative to the ship.
    :param ship_position: The cartesian (x, y) of the ship
    :param ship_heading: The heading of the ship, in radians. A zero-heading indicates the ship is facing right
                        Note that kessler-lib uses degrees, convert before calling.
    :param asteroid_positions: An (n,2) numpy array of the asteroid (x,y) positions
    :param map_size: A (2,) numpy array of the map size in (x, y) units
    :return: An (n,2) numpy array of the asteroid (rho, phi) positions relative to the ship.
             An angle of 0 indicates the asteroid is directly in front of the ship.
             The angle will always be within the range [0, 2pi)
             !! IMPORTANT!! The map wraps around at the edges. An asteroid may be 800 units "behind" the ship
                            (and look far away visually), but actually may be right in front of it due to wrapping.
                            There are, technically, infinitely many valid positions for each asteroid. This function
                            will only return the position with the smallest rho-value (i.e. closest to the ship).
    """
    # I'm not confident this is the most efficient solution! But seems to work.
    # Move the ship to the center of the map
    center = map_size / 2
    offset = center - ship_position
    ship_position = ship_position.copy() + offset

    # Offset everything by the same amount, and adjust anything that's now "out of bounds"
    centered_asteroids = np.mod(asteroid_positions + offset, map_size)

    # The ship is in the middle now, so the shortest path from ship <--> asteroid can never wrap around edges
    # Get the coordinates of asteroids relative to the ship (i.e. the center)
    centered_asteroids -= ship_position

    rho, phi = c2p(centered_asteroids[:, 0], centered_asteroids[:, 1])

    # Rotate everything relative to the ship's heading, keep in range [0, 2pi)
    phi -= ship_heading
    phi = np.mod(phi, 2 * np.pi)

    return np.stack([rho, phi], axis=-1)

def c2p(x, y):
    z = x + 1j * y
    return np.abs(z), np.angle(z)

def p2c(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)

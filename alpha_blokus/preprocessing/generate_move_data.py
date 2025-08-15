import os
import numpy as np
from tqdm import tqdm


def run(board_size, output_dir):
    BOARD_SIZE = board_size
    PRECOMPUTE_DIRECTORY = output_dir
    PIECES = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)],
        [(1, 1), (0, 1), (1, 0), (2, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (1, 0), (2, 0), (0, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 0)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0)],
    ]

    output = {}

    print("Populating values...")

    output["new_occupieds"] = []
    output["rotation_mapping"] = []
    output["piece_indices"] = []
    output["piece_orientation_indices"] = []
    output["placement_x"] = []
    output["placement_y"] = []
    output["center_placement_x"] = []
    output["center_placement_y"] = []
    output["new_corners"] = []
    output["new_adjacents"] = []

    piece_orientations = set()

    for piece_index in tqdm(range(len(PIECES))):
        occupieds_for_piece = []
        piece_indices = []
        piece_orientation_indices = []
        placement_x = []
        placement_y = []
        center_placement_x = []
        center_placement_y = []

        for rotation in range(4):
            for flip in range(2):
                piece = PIECES[piece_index][:]

                for _ in range(rotation + 1):
                    piece = [(y, -x) for x, y in piece]

                # Apply the flip to the piece
                if flip:
                    piece = [(x, -y) for x, y in piece]

                # Find the bottom left of the piece now.
                min_x = min(x for x, y in piece)
                min_y = min(y for x, y in piece)

                # Sort so the same piece with squares in different orders
                # is considered the same.
                piece.sort()

                # Shift the piece so that the bottom left corner is at (0, 0)
                piece = [(x - min_x, y - min_y) for x, y in piece]

                # Skip if we've already done this piece orientation.
                piece_hash = tuple(piece)
                if piece_hash in piece_orientations:
                    continue
                piece_orientations.add(piece_hash)

                # Find the maximum x and y values of the piece
                max_x = max(x for x, y in piece)
                max_y = max(y for x, y in piece)

                # Find the center of the piece relative to the origin (bottom left)
                piece_center_x = max_x // 2
                piece_center_y = max_y // 2

                for x in range(BOARD_SIZE - max_x):
                    for y in range(BOARD_SIZE - max_y):
                        occupieds_for_piece.append(
                            np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
                        )
                        piece_orientation_indices.append(len(piece_orientations) - 1)
                        placement_x.append(x)
                        placement_y.append(y)

                        center_x = x + piece_center_x
                        center_y = y + piece_center_y

                        assert 0 <= center_x < BOARD_SIZE
                        assert 0 <= center_y < BOARD_SIZE

                        center_placement_x.append(center_x)
                        center_placement_y.append(center_y)

                        piece_indices.append(piece_index)
                        for piece_x, piece_y in piece:
                            occupieds_for_piece[-1][x + piece_x][y + piece_y] = True

        occupieds_for_piece, indices = np.unique(
            np.array(occupieds_for_piece, dtype=bool), return_index=True, axis=0
        )
        output["new_occupieds"].append(occupieds_for_piece)
        output["piece_orientation_indices"].append(
            np.array(piece_orientation_indices)[indices]
        )
        output["placement_x"].append(np.array(placement_x)[indices])
        output["placement_y"].append(np.array(placement_y)[indices])
        output["center_placement_x"].append(np.array(center_placement_x)[indices])
        output["center_placement_y"].append(np.array(center_placement_y)[indices])
        output["piece_indices"].append(np.array(piece_indices)[indices])

        # Now, compute an array which maps each move to which move it would be if rotated 90 degrees.
        rotated_occupieds_for_piece = np.rot90(occupieds_for_piece, axes=(1, 2))
        comparison = np.equal(
            occupieds_for_piece[:, np.newaxis, :, :], rotated_occupieds_for_piece
        )
        assert comparison.shape == (
            occupieds_for_piece.shape[0],
            occupieds_for_piece.shape[0],
            BOARD_SIZE,
            BOARD_SIZE,
        )

        comparison_result = comparison.all(axis=(2, 3))

        # Assert that each row and column of comparison result has exactly one True value.
        assert np.all(np.sum(comparison_result, axis=0) == 1)
        assert np.all(np.sum(comparison_result, axis=1) == 1)

        # Return the index of the True value in each row and column as the remapping.
        remapping = np.argmax(comparison_result, axis=0)
        assert remapping.shape == (occupieds_for_piece.shape[0],)

        output["rotation_mapping"].append(remapping)

        for i, new_occupieds in enumerate(occupieds_for_piece):
            new_corners = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=bool)
            new_adjacents = np.zeros((1, BOARD_SIZE, BOARD_SIZE), dtype=bool)

            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    # Populate new corners
                    for x_to_check in [x + 1, x - 1]:
                        for y_to_check in [y + 1, y - 1]:
                            if (
                                x_to_check < 0
                                or x_to_check >= BOARD_SIZE
                                or y_to_check < 0
                                or y_to_check >= BOARD_SIZE
                            ):
                                continue
                            if new_occupieds[x_to_check, y_to_check]:
                                new_corners[0][x, y] = True

                    # Populate new adjacents
                    for x_to_check, y_to_check in [
                        (x + 1, y),
                        (x - 1, y),
                        (x, y + 1),
                        (x, y - 1),
                    ]:
                        if (
                            x_to_check < 0
                            or x_to_check >= BOARD_SIZE
                            or y_to_check < 0
                            or y_to_check >= BOARD_SIZE
                        ):
                            continue
                        if new_occupieds[x_to_check, y_to_check]:
                            new_adjacents[0][x, y] = True

            output["new_corners"].append(new_corners)
            output["new_adjacents"].append(new_adjacents)

    output["new_occupieds"] = np.concatenate(output["new_occupieds"])
    output["piece_indices"] = np.concatenate(output["piece_indices"])
    output["piece_orientation_indices"] = np.concatenate(
        output["piece_orientation_indices"]
    )
    output["placement_x"] = np.concatenate(output["placement_x"])
    output["placement_y"] = np.concatenate(output["placement_y"])
    output["center_placement_x"] = np.concatenate(output["center_placement_x"])
    output["center_placement_y"] = np.concatenate(output["center_placement_y"])

    NUM_MOVES = output["new_occupieds"].shape[0]

    rows_so_far = 0
    for rm in output["rotation_mapping"]:
        rm += rows_so_far
        rows_so_far += len(rm)

    rotation_mapping = np.concatenate(output["rotation_mapping"])
    assert rotation_mapping.shape == (NUM_MOVES,)
    assert (np.sort(rotation_mapping) == np.arange(NUM_MOVES)).all()

    output["rotation_mapping"] = np.zeros((4, NUM_MOVES), dtype=int)
    output["rotation_mapping"][0] = np.arange(NUM_MOVES)
    output["rotation_mapping"][1] = rotation_mapping
    output["rotation_mapping"][2] = rotation_mapping[rotation_mapping]
    output["rotation_mapping"][3] = rotation_mapping[rotation_mapping[rotation_mapping]]

    output["scores"] = np.sum(output["new_occupieds"], axis=(1, 2), dtype=int)
    assert output["scores"].shape == (NUM_MOVES,)
    output["piece_indices"] = np.array(output["piece_indices"], dtype=int)
    output["piece_orientation_indices"] = np.array(
        output["piece_orientation_indices"], dtype=int
    )
    output["placement_x"] = np.array(output["placement_x"], dtype=int)
    output["placement_y"] = np.array(output["placement_y"], dtype=int)
    output["new_corners"] = np.concatenate(output["new_corners"])
    output["new_adjacents"] = np.concatenate(output["new_adjacents"])

    print("Done populating moves dictionary.")

    print(f"Generated {NUM_MOVES} moves.")

    print("Computing moves using same piece...")
    moves_using_same_piece = (
        output["piece_indices"][:, np.newaxis] == output["piece_indices"]
    )
    assert moves_using_same_piece.shape == (NUM_MOVES, NUM_MOVES)

    def batch_and(a1, a2):
        batch_size = 2000
        result = np.zeros((NUM_MOVES, NUM_MOVES), dtype=bool)
        for i in tqdm(range(0, NUM_MOVES, batch_size), leave=False):
            end_i = min(i + batch_size, NUM_MOVES)
            new_a1_batch = a1[i:end_i]  # Shape: (batch_size_i, 20, 20)
            new_a1_batch_expanded = new_a1_batch[
                :, np.newaxis, :, :
            ]  # Shape: (batch_size_i, 1, 20, 20)

            for j in tqdm(range(0, NUM_MOVES, batch_size), leave=False):
                end_j = min(j + batch_size, NUM_MOVES)
                new_a2_batch = a2[j:end_j]  # Shape: (batch_size_j, 20, 20)

                # Compute the boolean operation in the current batch
                temp_result = (new_a1_batch_expanded & new_a2_batch).any(axis=(2, 3))
                result[i:end_i, j:end_j] = temp_result

        return result

    print("Computing moves occupying new adjacents...")
    moves_occupying_new_adjacents = batch_and(
        output["new_adjacents"], output["new_occupieds"]
    )
    assert moves_occupying_new_adjacents.shape == (NUM_MOVES, NUM_MOVES)

    print("Computing moves occupying new occupieds...")
    moves_occupying_new_occupieds = batch_and(
        output["new_occupieds"], output["new_occupieds"]
    )
    assert moves_occupying_new_adjacents.shape == (NUM_MOVES, NUM_MOVES)

    print("Computing moves occupying new corners...")
    moves_occupying_new_corners = batch_and(
        output["new_corners"], output["new_occupieds"]
    )
    assert moves_occupying_new_corners.shape == (NUM_MOVES, NUM_MOVES)

    print("Computing output values...")

    output["moves_ruled_out_for_player"] = (
        moves_using_same_piece | moves_occupying_new_adjacents
    )
    del moves_using_same_piece
    del moves_occupying_new_adjacents

    output["moves_ruled_out_for_all"] = moves_occupying_new_occupieds
    output["moves_enabled_for_player"] = moves_occupying_new_corners

    print("Saving outputs to disk...")
    os.makedirs(PRECOMPUTE_DIRECTORY, exist_ok=True)
    np.savez_compressed(f"{PRECOMPUTE_DIRECTORY}/moves.npz", **output)

    print("Done.")


if __name__ == "__main__":
    run(20, "/tmp/alpha_blokus_moves/")

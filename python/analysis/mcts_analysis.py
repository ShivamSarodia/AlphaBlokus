def array_index_top_k(iterable, array_index_to_move_index, k=10):
    top_k_enumerated = sorted(enumerate(iterable), key=lambda x: x[1], reverse=True)[:k]
    top_k_array_indices = [array_index for array_index, _ in top_k_enumerated]
    top_k_move_indices = [
        array_index_to_move_index[array_index] for array_index in top_k_array_indices
    ]
    top_k_values = [iterable[array_index] for array_index in top_k_array_indices]
    return list(zip(top_k_array_indices, top_k_move_indices, top_k_values))

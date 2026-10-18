import random
from functools import reduce


def merge_dicts_of_lists(*dict_lst: dict) -> dict:
    def inner_merge_dicts_of_lists(dict1: dict, dict2: dict):
        inner_result = dict1.copy()
        for key, value in dict2.items():
            if key in inner_result:
                if isinstance(inner_result[key], dict):
                    inner_result[key] = inner_merge_dicts_of_lists(inner_result[key], value)
                elif isinstance(inner_result[key], list):
                    inner_result[key] += value
                else:
                    raise Exception("Invalid dct value type!")
            else:
                inner_result[key] = value.copy()
        return inner_result

    result = reduce(inner_merge_dicts_of_lists, dict_lst)
    return result


def _flatten_dict_of_lsts(dct):
    """
    Flatten a nested dict into a list of dicts.
    Input example:
    {
        'a': {
            'b': [1, 2, 3],
            'c': [4, 5, 6],
        }
    }

    Output:
    [
        (('a', 'b'), 1),
        (('a', 'b'), 2),
        (('a', 'b'), 3),
        (('a', 'c'), 4),
        (('a', 'c'), 5),
        (('a', 'c'), 6),
    ]
    """
    assert len({type(v) for v in dct.values()}) == 1, "dictionary values have multiple types!"
    assert all(isinstance(v, list | dict) for v in dct.values()), "dictionary values have non dict/list types!"

    ks, vs = list(dct.keys()), list(dct.values())
    if isinstance(vs[0], dict):
        results = []
        for k, v in zip(ks, vs):
            flatten_inner_dct = _flatten_dict_of_lsts(v)
            results.extend([((k, *inner_k_tup), inner_v) for inner_k_tup, inner_v in flatten_inner_dct])
        return results
    else:
        return [((k,), v) for k, v_lst in dct.items() for v in v_lst]


def _restore_dict_of_lsts_from_flatten(flatten_dct_lst):
    """
    Restore dict from a flattened list of dicts.

    Input example:
    [
        (('a', 'b'), 1),
        (('a', 'b'), 2),
        (('a', 'b'), 3),
        (('a', 'c'), 4),
        (('a', 'c'), 5),
        (('a', 'c'), 6),
    ]

    Output:
    {
        'a': {
            'b': [1, 2, 3],
            'c': [4, 5, 6],
        }
    }
    """
    ks = [tup[0] for tup in flatten_dct_lst]
    vs = [tup[1] for tup in flatten_dct_lst]
    if len(ks[0]) == 0:
        return vs

    result = {k[0]: [] for k in ks}
    for k, v in zip(ks, vs):
        result[k[0]].append((k[1:], v))

    return {k: _restore_dict_of_lsts_from_flatten(v) for k, v in result.items()}


def sample_dict_across_sublists(
    lsts: dict[str, list], n_sample: int, exact: bool = True, random_state: random.Random | None = None
) -> dict[str, list]:
    """
    Sample items from a dictionary of lists, return dictionary with similar keys mapped to the list of sampled values.
    Sampling is done globally, not per key.
    If exact=True, sample exactly n_sample elements; otherwise sample between 1 and n_sample elements.
    If random_state is provided, it is used for sampling; otherwise the module-level random is used.
    """
    if n_sample == 0:
        return lsts
    if n_sample < 1:
        raise ValueError("n_sample should be positive!")

    # Flatten with indices to track source list
    indexed_elements = _flatten_dict_of_lsts(lsts)

    if exact and n_sample > len(indexed_elements):
        raise ValueError(f"not enough elements to sample - {n_sample=}, {len(indexed_elements)=}")
    n_sample = min(len(indexed_elements), n_sample)

    rng = random_state if random_state is not None else random

    # Determine actual sample size
    actual_sample = n_sample if exact else rng.randint(1, n_sample)

    # Sample actual_sample elements
    sampled = rng.sample(indexed_elements, actual_sample)

    # Organize sampled items back into sublists
    result = _restore_dict_of_lsts_from_flatten(sampled)

    # Delete empty items
    result = {k: v_lst for k, v_lst in result.items() if len(v_lst) > 0}

    return result

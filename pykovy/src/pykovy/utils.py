from bisect import bisect_right
from collections import deque, defaultdict
from functools import update_wrapper
from itertools import islice, accumulate
from operator import itemgetter
from random import choice, random


__all__ = (
    "window", "weighted_choice", "unzip",
    "patch_return_type", "weighted_choice_on_map", "random_key",
    "head", "last", "groupby"
)


def window(it, size):
    """Returns a sliding window (of width size) over data from the iterable.

    This is a spin on a recipe provided in the 2.3.5 itertools docs, using
    a deque rather than playing surgeon on a tuple.
    """

    it = iter(it)
    window = deque(islice(it, size), maxlen=size)
    yield tuple(window)
    for i in it:
        window.append(i)
        yield tuple(window)


def weighted_choice(weights):
    """Creates a list of running totals for weights to choose from
    and returns a callable to handle the actual choosing process.

    This closure is a compromise between an object maintaining state and
    recreating the running total every time we need to choose an element.

    The closure allows deterministic responses for testing purposes
    instead of inherently relying on random.random(), though that's
    likely the only actual input.
    """

    weights = list(accumulate(weights))

    def weighted_chooser(choice):
        choice *= weights[-1]
        return bisect_right(weights, choice)

    return weighted_chooser


def weighted_choice_on_map(mapping, randomizer=random):
    """Creates a weighted choice closure on a mapping. It uses the values
    as the weights and the keys as the final choices.

    Like :func:`weighted_choice` this returns a closure. And it also
    allows passing in a function to return floats 0 <= n < 1 if random.random
    should not be used.
    """

    values, chances = unzip(sorted(mapping.items(), key=itemgetter(1)))
    chooser = weighted_choice(chances)

    def random_item():
        """Closure to associate indices return by weighted_choices
        with indices of actual values.
        """

        choice = chooser(randomizer())
        return values[choice]

    return random_item


def unzip(groups):
    """Simple wrapper around zip(*groups). Providing a name is a lot clearer
    than having the raw code around and someone wondering what it's doing.

    [(1,'a'), (2, 'b'), (3, 'c')] -> [(1,2,3), ('a', 'b', 'c')]
    """

    return zip(*groups)


def random_key(mapping, chooser=choice):
    """Returns a random key from a mapping.

    Allows passing a function that selects a random item from a list, but
    defaults to random.choice. Given the nature of dictionary keys, this
    function is inherently non-deterministic.
    """

    return chooser(list(mapping.keys()))


def _coerce_return(name, source_cls):
    """Helper to coerce return type on methods that explicitly create a type
    rather than relying on reflection to determine the appropriate type.
    """
    old = getattr(source_cls, name)
    return update_wrapper(lambda s, *a, **k: s.__class__(old(s, *a, **k)), old)


def patch_return_type(names, source_cls):
    """Class decorator to coerce return types on methods that don't rely on
    reflection to determine the appropriate return type.

        ..code-block:: python
        @patch_return_type(['__add__'], Counter)
        class MyCounter(Counter):
            pass

        type(MyCounter('a') + MyCounter('a')) #MyCounter rather than Counter
    """

    def patcher(cls):
        for name in names:
            setattr(cls, name, _coerce_return(name, source_cls))
        return cls
    return patcher


# shamelessly stolen from toolz.itertoolz
def groupby(seq, key):
    """Groups an iterable based on a key function.
    """
    d = defaultdict(lambda: [].append)

    for item in seq:
        d[key(item)](item)

    return {k: v.__self__ for k, v in d.items()}


head = itemgetter(slice(-1))
last = itemgetter(-1)

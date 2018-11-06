from collections import Counter, Mapping, MutableMapping
from itertools import chain
from random import random
from .errors import MarkovError, DisjointChainError, MarkovStateError
from .utils import (
    window, weighted_choice_on_map,
    patch_return_type, random_key,
    head, last, groupby
)

__all__ = (
    "MarkovChain", "MarkovChainIterator", "ProbablityMap"
)

# these methods in counter explicitly return
# a new instance of Counter rather than
# returning an instance of the calling class
__COUNTER_OVERRIDE_METHODS = [
    '__add__', '__sub__', '__or__',
    '__and__', '__pos__', '__neg__',
]


@patch_return_type(__COUNTER_OVERRIDE_METHODS, Counter)
class ProbablityMap(Counter):
    """Specialized Counter class that provides for creating a weighted
    random choice from it's key-value pairs.

    Certain methods from Counter are overriden to provide for correct
    return types (ProbablityMap instead of Counter):
        * __add__
        * __sub__
        * __or__
        * __and__
        * __pos__
        * __neg__

    These overrides are handled by the patch_return_type decorator.

    If ProbablityMap is inherited from, the return type of these methods
    will be of the subclass rather than ProbablityMap.
    """

    def weighted_choice(self, randomizer=random):
        """Returns a closure to allow pulling weighted, random keys
        from the object based on the "counted" value.

        Allows passing a function that returns floats 0 <= n < 1
        if random.random should not be used.

        The closure is frozen to the state of the object when it was called.
        """

        return weighted_choice_on_map(self, randomizer)


class MarkovChain(MutableMapping):
    """A collection of states and possible states.

    States are keys stored as n-length tuples and possible states are values
    stored as ProbablityMap instances.
    """

    def __init__(self, order, states=None):
        self.data = {}
        self._order = order
        if states is not None:
            self.update(states)

    def __repr__(self):
        return "{}(order={})".format(self.__class__.__name__, self.order)

    @property
    def order(self):
        """Order of the chain, refers to length of keys"""
        return self._order

    @order.setter
    def order(self, value):
        raise TypeError("{}.order is read only".format(self.__class__.__name__))

    # equality is more like "compatibility" meaning two MarkovChains
    # have the same order and could be merged
    # this does not imply that *any* shared states exist, however
    def __eq__(self, other):
        return isinstance(other, MarkovChain) and self.order == other.order

    def __iter__(self):
        """Return a default MarkovChainIterator.

        If optional arguments needs to be passed to iterator,
        use :method:`MarkovChain.iterate_chain`
        """
        return MarkovChainIterator(chain=self.data)

    def __setitem__(self, key, value):
        """Sets key-value pair on the MarkovChain and ensures type saftey
        of keys and values.
        """
        if not isinstance(key, tuple) or len(key) != self.order:
            raise TypeError("key must be tuple of length {}".format(self.order))

        if not isinstance(value, Mapping):
            raise TypeError("value must be a mapping")

        if not isinstance(value, ProbablityMap):
            value = ProbablityMap(value)

        self.data[key] = value

    def __getitem__(self, state):
        if not isinstance(state, tuple):
            state = (state,)
        return self.data[state]

    def __len__(self):
        return len(self.data)

    def __delitem__(self, key):
        raise MarkovError(
            "Cannot delete from probablity chain without possibly "
            "becoming disjoint. If you meant this, use "
            "{}.data.__delitem__".format(self.__class__.__name__)
        )

    def __contains__(self, v):
        return v in self.data

    def update(self, other=None, **kwds):
        """Updates the chain from a mapping/iterable.
        """
        other = other or ()
        if isinstance(other, Mapping):
            for key in other:
                self[key] = other[key]
        elif hasattr(other, 'keys'):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value

        for key, value in kwds.items():
            self[key] = value

    # proxy keys, values and items to underlying dictionary
    # to avoid accidentally creating MarkovChainIterator instances
    # which is likely *not* the desired behavior
    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    @classmethod
    def from_corpus(cls, corpus, order, begin_with=None):
        """Allows building a Markov Chain from a corpus rather than
        constructing it iteratively by hand.

        Corpus may be any iterable. If it is a string and words are
        intended to be preserved, use str.split or similar to convert
        it to a list before hand.

        * corpus: iterable of objects to build the chain over
        * order: order of the new chain
        * begin_with: allows preprending placeholder values
        before the actual content of the corpus to serve as a
        known starting point if needed.
        """

        if begin_with is not None:
            corpus = chain(begin_with, corpus)

        groups = groupby(window(corpus, size=order + 1), head)
        staging = {
            k: ProbablityMap(map(last, v))
            for k, v in groups.items()
        }

        return cls(states=staging, order=order)

    def iterate_chain(self, **kwargs):
        """Allows passing arbitrary keyword arguments to the
        MarkovChainIterator class for iteration.
        """

        return MarkovChainIterator(chain=self.data, **kwargs)


class MarkovChainIterator(object):
    """Iteration handler for MarkovChains.

    Maintains a copy of the original chain's keyed states and creates
    a weighted random choice closure from keys' possible states
    to prevent interference when modifying the original chain during iteration.

    It should be noted that this is a non-deterministic, possibly cyclical
    iterator.
    """

    def __init__(self, chain, randomizer=random, begin_at=None, **kwargs):
        """Set initial state of the iterator.

        * chain: MarkovChain or subclass to iterate
        * randomizer: callable that returns floats 0 < n < 1,
        defaults to :func:`~random.random`
        * begin_at: known state to place the iterator in
        * randomizer: function to generate floats 0 <= n < 1
        defaults to random.random
        """

        self._invalid = False
        self._state = self._possible = None
        self._randomizer = randomizer
        self._chain = self._build_chain(chain)

        if begin_at:
            self._set_state(begin_at)
        else:
            self._random_state()

    def reset(self, begin_at=None, **kwargs):
        """Places the iterator back into a known or random state.

        Useful for reusing the same iterator multiple times without having
        to rebuild every weighted random chooser.

        * begin_at: known state to place the iterator in
        """
        self._invalid = False
        self._state = self._possible = None

        if begin_at:
            self._set_state(begin_at)
        else:
            self._random_state()

    def _set_state(self, begin_at=None):
        """Attempts to place iterator into a known state and falls back
        to a random state if the known state isn't possible.
        """

        try:
            self.state = begin_at
        except MarkovStateError:
            self._random_state()

    def _build_chain(self, chain):
        """Builds map of states and weighted random
        closures from a Markov Chain's possible states.
        """

        return {
            state: chain[state].weighted_choice(self._randomizer)
            for state in chain
        }

    def _random_state(self):
        "Puts the chain into a random state."

        self.state = random_key(self._chain)

    @property
    def state(self):
        """Current state of the iterator.

        If set, the iterator tries to enter a known state. If the known
        state does not exist, a MarkovStateError is raised.
        """

        return self._state

    @state.setter
    def state(self, state):
        """Attempts to set the iterator in a known state.

        If that isn't possible, raises a MarkovStateError.
        """

        if state not in self._chain:
            raise MarkovStateError("Invalid state provided: {}".format(state))

        self._state = state
        self._possible = self._chain[state]

    def __iter__(self):
        return self

    def __next__(self):
        """Steps through states until an invalid state is reached,
        which stops iteration with a DisjointChainError.
        """
        if self._invalid:
            raise DisjointChainError(self._invalid)

        value = self._possible()

        try:
            self.state = self._state[1:] + tuple([value])
        except MarkovStateError as e:
            self._invalid = e

        return value

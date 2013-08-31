import numpy as np
from collections import OrderedDict


class StructuredArray(object):
    """An array of different blocks with equal first dimension. """

    def __init__(self, blocks):
        shape_0 = set(b.shape[0] for b in blocks.values())
        if len(shape_0) != 1:
            raise ValueError('all blocks must agree on first dim')

        self.__dict__['shape'] = (tuple(shape_0)[0], len(blocks))
        self.__dict__['blocks'] = blocks

    def __getitem__(self, slz):
        """Get item ``slz`` of structured array.

        Parameters
        ----------
        slz : int or slice or str
            Either the slice object or int used to slice each block
            or a str to access a specific block.

        Returns
        -------
        res : StructuredArray or block
            A structured array with sliced blocks or the selected block.

        Example
        -------
        >>> x = StructuredArray({'foo': np.arange(10)})
        >>> x['foo']
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> x[0].foo
        array([0])
        """

        if isinstance(slz, basestring):
            # block access
            block = self.blocks[slz]
            return block
        else:
            # array slicing
            blocks = OrderedDict((n, np.atleast_1d(a[slz]))
                                 for n, a in self.blocks.iteritems())
            attrs = [(key, val) for key, val in self.__dict__.iteritems()
                     if ((not (key.startswith('__') and key.endswith('__')))
                         and key not in ('blocks', 'shape'))]
            new_sa = StructuredArray(blocks)
            for key, val in attrs:
                new_sa.__dict__[key] = val
            return new_sa

    def __setitem__(self, key, val):
        if val.shape[0] != self.shape[0]:
            raise ValueError('first dimension do not agree '
                             '({0} != {1})'.format(val.shape[0], self.shape[0]))
        self.blocks[key] = val

    def __getattr__(self, attr):
        # needed otherwise pickle breaks
        if attr.startswith('__') and attr.endswith('__'):
            return super(StructuredArray, self).__getattr__(attr)
        elif attr in self.blocks:
            return self.blocks[attr]
        else:
            raise AttributeError('%s not found' % attr)

    def __setattr__(self, attr, val):
        if attr in self.blocks:
            self[attr] = val
        else:
            self.__dict__[attr] = val

    def values(self):
        """Returns a numpy array holding concatenation of all blocks. """
        blocks = []
        for b in self.blocks.itervalues():
            if b.ndim == 1:
                b = b[:, np.newaxis]
            blocks.append(b)
        res = np.hstack(blocks)
        return res

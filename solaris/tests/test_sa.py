import numpy as np

from solaris import sa


def test_attr_set():
    x = sa.StructuredArray({'foo': np.random.rand(10, 1)})
    x.foo = np.arange(10)
    x['foo'] = np.arange(10)[::-1]
    np.testing.assert_array_equal(x.foo, np.arange(10)[::-1])


def test_getitem():
    x = sa.StructuredArray({'foo': np.arange(10)})
    x.name = 'foobar'
    y = x[0]
    assert isinstance(y, sa.StructuredArray)
    assert y.shape == (1, 1)
    assert y.foo[0] == 0
    assert y.name == 'foobar'


def test_getitem_block():
    x = sa.StructuredArray({'foo': np.arange(10)})
    true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(x.foo, true)
    np.testing.assert_array_equal(x['foo'], true)


def test_getslice():
    x = sa.StructuredArray({'foo': np.arange(10)})
    x.name = 'foobar'
    y = x[0:5]
    assert isinstance(y, sa.StructuredArray)
    assert y.shape == (5, 1)
    np.testing.assert_array_equal(y.foo, np.arange(5))
    assert y.name == 'foobar'


def test_values():
    x = sa.StructuredArray({'foo': np.arange(10),
                            'bar': np.arange(10),})
    vals = x.values()
    assert vals.shape == (10, 2)


def test_pickle():
    x = sa.StructuredArray({'foo': np.arange(10),
                            'bar': np.arange(10),})
    x.name = 'barfoo'
    from sklearn.externals import joblib
    joblib.dump(x, '/tmp/foobar.pkl')
    x_ = joblib.load('/tmp/foobar.pkl')

    np.testing.assert_array_equal(x_.foo, x.foo)
    np.testing.assert_array_equal(x_.bar, x.bar)
    assert x_.name == x.name


def test_obj_attr():
    x = sa.StructuredArray({'foo': np.arange(10),
                            'bar': np.arange(10),})
    x.name = 'foobar'
    assert x.name == 'foobar'
    assert 'name' not in x.blocks

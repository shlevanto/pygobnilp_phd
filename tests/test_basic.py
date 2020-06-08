from pygobnilp.gobnilp import Gobnilp
m = Gobnilp()

def test_bge():
    m.learn('data/gaussian.dat',data_type='continuous',score='BGe',plot=False,palim=None)
    assert abs(m.learned_scores[0] - -53258.94161814058) < 0.0001

def test_bic():
    m.learn('data/gaussian.dat',data_type='continuous',score='GaussianBIC',plot=False,palim=None)
    assert abs(m.learned_scores[0] - -53221.34568733569) < 0.0001
    m.learn('data/gaussian.dat',data_type='continuous',score='GaussianBIC',plot=False,palim=None,sdresidparam=False)
    assert abs(m.learned_scores[0] - -53191.53551116573) < 0.0001

def test_bdeu():
    m.learn('data/discrete.dat',plot=False,palim=None)
    assert abs(m.learned_scores[0] - -24028.0947783535) < 0.0001

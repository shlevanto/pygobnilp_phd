from pygobnilp.gobnilp import Gobnilp
m = Gobnilp()

epsilon = 0.0001
def optimal_score_is(m,score):
    return abs(m.learned_scores[0] - score) < epsilon

def test_bge():
    m.learn('data/gaussian.dat',data_type='continuous',score='BGe',plot=False,palim=None)
    assert optimal_score_is(m, -53258.94161814058)

def test_bic():
    m.learn('data/gaussian.dat',data_type='continuous',score='GaussianBIC',plot=False,palim=None)
    assert optimal_score_is(m, -53221.34568733569)
    m.learn('data/gaussian.dat',data_type='continuous',score='GaussianBIC',plot=False,palim=None,sdresidparam=False)
    assert optimal_score_is(m, -53191.53551116573)

def test_bdeu():
    m.learn('data/discrete.dat',plot=False,palim=None)
    assert optimal_score_is(m, -24028.0947783535)

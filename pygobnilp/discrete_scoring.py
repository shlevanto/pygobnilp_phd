#!/usr/bin/env python
#/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *   GOBNILP (Python version) Copyright (C) 2019 James Cussens           *
# *                                                                       *
# *   This program is free software; you can redistribute it and/or       *
# *   modify it under the terms of the GNU General Public License as      *
# *   published by the Free Software Foundation; either version 3 of the  *
# *   License, or (at your option) any later version.                     *
# *                                                                       *
# *   This program is distributed in the hope that it will be useful,     *
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of      *
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU    *
# *   General Public License for more details.                            *
# *                                                                       *
# *   You should have received a copy of the GNU General Public License   *
# *   along with this program; if not, see                                *
# *   <http://www.gnu.org/licenses>.                                      *
"""
    Code for computing local scores from discrete data
"""

__author__ = "Josh Neil, James Cussens"
__email__ = "james.cussens@york.ac.uk"

from math import lgamma, log

import numpy as np

from itertools import combinations

import sys

from numba import jit, njit

import argparse

from scipy.special import digamma

import pandas as pd

# START functions for contabs

@jit(nopython=True)
def marginalise_uniques_contab(unique_insts, counts, cols):
    '''
    Marginalise a contingency table represented by unique insts and counts
    '''
    marg_uniqs, indices = np.unique(unique_insts[:,cols], return_inverse=True)
    marg_counts = np.zeros(len(marg_uniqs),dtype=np.uint32)
    for i in range(len(counts)):
        marg_counts[indices[i]] += counts[i]
    return marg_uniqs, marg_counts

@jit(nopython=True)
def make_contab(data, counts, cols, arities, maxsize):
    '''
    Compute a marginal contingency table from data or report
    that the desired contingency table would be too big.

    All inputs except the last are arrays of unsigned integers

    Args:
     data (numpy array): the unique datapoints as a 2-d array, each row is a datapoint, assumed unique
     counts (numpy array): the count of how often each unique datapoint occurs in the original data
     cols (numpy array): the columns (=variables) for the marginal contingency table.
      columns must be ordered low to high
     arities (numpy array): the arities of the variables (=columns) for the contingency table
      order must match that of `cols`.
     maxsize (int): the maximum size (number of cells) allowed for a contingency table

    Returns:
     tuple: 1st element is of type ndarray: 
      If the contingency table would have more elements than `maxsize' then the array is empty
      (and the 2nd element of the tuple should be ignored)
      else an array of counts of length equal to the product of the `arities`.
      Counts are in lexicographic order of the joint instantiations of the columns (=variables)
      2nd element: the 'strides' for each column (=variable)
    '''
    p = len(cols)
    #if arities = (2,3,3) then strides = 9,3,1
    #if row is (2,1,2) then index is 2*9 + 1*3 + 2*1
    strides = np.empty(p,dtype=np.uint32)
    idx = p-1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[idx]
        if stride > maxsize:
            return np.empty(0,dtype=np.uint32), strides
        idx -= 1
    contab = np.zeros(stride,dtype=np.uint32)
    for rowidx in range(data.shape[0]):
        idx = 0
        for i in range(p):
            idx += data[rowidx,cols[i]]*strides[i]
        contab[idx] += counts[rowidx]
    return contab, strides

@jit(nopython=True)
def _compute_ll_from_flat_contab(contab,strides,child_idx,child_arity):
    child_stride = strides[child_idx]
    ll = 0.0
    child_counts = np.empty(child_arity,dtype=np.int32)
    contab_size = len(contab)
    for i in range(0,contab_size,child_arity*child_stride):
        for j in range(i,i+child_stride):
            n = 0
            for k in range(child_arity):
                count = contab[j]
                child_counts[k] = count
                n += count
                j += child_stride
            if n > 0:
                for c in child_counts:
                    if c > 0:
                        ll += c * log(c/n)
    return ll

@jit(nopython=True)
def _compute_ll_from_unique_contab(data,counts,n_uniqs,pa_idxs,child_arity,orig_child_col):
    child_counts = np.zeros((n_uniqs,np.int64(child_arity)),dtype=np.uint32)
    for i in range(len(data)):
        child_counts[pa_idxs[i],data[i,orig_child_col]] += counts[i]
    ll = 0.0
    for i in range(len(child_counts)):
        #n = child_counts[i,:].sum() #to consider
        n = 0
        for k in range(child_arity):
            n += child_counts[i,k]
        for k in range(child_arity):
            c = child_counts[i,k]
            if c > 0:
                ll += c * log(c/n)
    return ll

@jit(nopython=True)
def compute_bdeu_component(data, counts, cols, arities, alpha, maxflatcontabsize):
    contab = make_contab(data, counts, cols, arities, maxflatcontabsize)[0]
    if len(contab) > 0:
        alpha_div_arities = alpha / len(contab)
        non_zero_count = 0
        score = 0.0
        for count in contab:
            if count != 0:
                non_zero_count += 1
                score -= lgamma(alpha_div_arities+count) 
        score += non_zero_count*lgamma(alpha_div_arities)  
        return score, non_zero_count


# START functions for upper bounds

@jit(nopython=True)
def lg(n,x):
    return lgamma(n+x) - lgamma(x)

def h(counts):
    '''
    log P(counts|theta) where theta are MLE estimates computed from counts
    ''' 
    tot = sum(counts)
    #print(tot)
    res = 0.0
    for n in counts:
        if n > 0:
            res += n*log(n/tot)
    return res

def chisq(counts):
    tot = sum(counts)
    t = len(counts)
    mean = tot/t #Python 3 - this creates a float
    chisq = 0.0
    for n in counts:
        chisq += (n - mean)**2
    return chisq/mean

def hsum(distss):
    res = 0.0
    for dists in distss:
        res += sum([h(d) for d in dists])
    return res

@jit(nopython=True)
def fa(dist,sumdist,alpha,r):
    #res = -lg(sumdist,alpha)
    res = lgamma(alpha) - lgamma(sumdist+alpha)
    alphar = alpha/r
    k = 0
    for n in dist:
        if n > 0:
            #res += lg(n,alphar)
            #res += (lgamma(n+alphar) - lgamma(alphar))
            res += lgamma(n+alphar)
            k += 1
    return res - k*lgamma(alphar)

def diffa(dist,alpha,r):
    """Compute the derivative of local-local BDeu score

    numba does not support the scipy.special functions -- we are using the
    digamma function in defining the entropy of the Chisq
    distribution. For this end I added a python script which contains code
    for a @vectorize-d digamma function. If we need to use anything from
    scipy.special we will have to write it up ourselves.

    """
    args = [n+alpha/r for n in dist] + [alpha,sum(dist)+alpha,alpha/r]
    z = digamma(args)
    return sum(z[:r+1]) - z[r+1] - r*z[r+2] 

def onepositive(dist):
    """
    Is there only one positive count in `dist`?
    """
    npos = 0
    for n in dist:
        if n > 0:
            npos += 1
            if npos > 1:
                return False
    return True

@njit
def array_equal(a,b,pasize):
    for i in range(pasize):
        if a[i] != b[i]:
            return False
    return True

#@njit
#def get_elems(a,idxs):
#    return np.a

@njit
def upper_bound_james_fun(atoms_ints,atoms_floats,pasize,alpha,r,idxs):
    ub = 0.0
    local_ub = 0.0
    best_diff = 0.0
    pasize_r = pasize+r
    lr = -log(r)
    oldrow = atoms_ints[idxs[0]]
    for i in idxs:
        row = atoms_ints[i]
        
        if not array_equal(oldrow,row,pasize):
            ub += min(local_ub+best_diff,lr)
            best_diff = 0.0
            local_ub = 0.0
            oldrow = row

        mle = atoms_floats[i]
        local_ub += mle
        if row[-1]: # if chi-sq condition met
            diff = fa(row[pasize:pasize_r],row[pasize_r],alpha/2.0,r) - mle
            if diff < best_diff:
                best_diff = diff
    ub += min(local_ub+best_diff,lr)
    return ub


#@jit(nopython=True)
def ub(dists,alpha,r):
    '''
    Args:
        dists (iter): list of (child-value-counts,chisqtest,mles) lists, for some
          particular instantiation ('inst1') of the current parents.
          There is a child-value-counts list for each non-zero instantiation ('inst2') of the biggest
          possible superset of the current set of parents. inst1 and each inst2 have the same
          values for the current parents.
        alpha (float): ESS/(num of current parent insts)
        r (int): arity of the child

    Returns:
       float: An upper bound on the BDeu local score for (implicit) child
        over all possible proper supersets of the current parents
    '''
    naives = 0.0
    best_diff = 0.0
    for (dist,sumdist,ok_first,naive_ub) in dists:
        naives += naive_ub
        #iffirst_ub = naive_ub
        #iffirst_ub = min(len(dist)*-log(r),naive_ub)
        if ok_first:
            diff = fa(dist,sumdist,alpha/2.0,r) - naive_ub
            #iffirst_ub = min(iffirst_ub,fa(dist,sumdist,alpha/2.0,r))
            #diff = iffirst_ub - naive_ub
            if diff < best_diff:
                best_diff = diff
    return best_diff + naives


# @jit(nopython=True)
# def _ll_score(contab_generator, arities, variables, child, process_contab_function):
#     """
#         Parameters:

#         - `contab_generator`: Function used to generate the contingency tables for `variables`
#         - `arities`: A NumPy array containing the arity of every variable
#         - `variables`: A Numpy array of ints which are column indices for the family = (child + parents)
#         - `child`: The column index for the child 
#         - `process_contab_function`: Function called to do main computation

#         - Returns the fitted log-likelihood (LL) score 
#     """
#     contab = contab_generator.make_contab(variables)
#     score, non_zero_count = process_contab_function(arities, contab, variables, alpha_div_arities)             
    
#     return score, non_zero_count



#@jit(nopython=True)
def get_atoms(data,arities):
    '''
    Args: 
        data(np.array): Discrete data as a 2d array of ints

    Returns:
        list: a list `atoms`, where `atoms[i]` is a dictionary mapping instantations
         of variables other than i to a tuple with 3 elements:

          1. the child counts (n1, n2, .., nr) for that instantations
          2. n * the entropy for the empirical distribution (n1/n, n2/n, .., nr/n)
          3. Whether the chi-squared statistic for (n1, n2, .., nr) exceeds r-1 
        
        Chi-squared test comes from "Compound Multinomial Likelihood Functions are Unimodal: 
        Proof of a Conjecture of I. J. Good". Author(s): Bruce Levin and James Reeds
        Source: The Annals of Statistics, Vol. 5, No. 1 (Jan., 1977), pp. 79-87
    
        At least two of the ni must be positive for the inst-tuple pair to be included
        in the dictionary since only in that case is the inst-tuple useful for getting
        a good upper bound.
       
    '''
    fullinsts = []
    for i in range(data.shape[1]):
        fullinsts.append({})
    for row in data:
        row = tuple(row)
        for i, val in enumerate(row):
            fullinsts[i].setdefault(row[:i]+(0,)+row[i+1:],  # add dummy value for ith val
                                    [0]*arities[i])[val] += 1

    # now, for each child i, delete full insts which are 'deterministic'
    # i.e. where only one child value is non-zero
    newdkts_ints = []
    newdkts_floats = []
    for i, dkt in enumerate(fullinsts):
        #print('old',len(dkt))
        #newdkt = {}
        newdkt_ints = []
        newdkt_floats = []
        th = arities[i] - 1
        for inst, childcounts in dkt.items():
            if len([v for v in childcounts if v > 0]) > 1:
                newdkt_ints.append(inst+tuple(childcounts)+(sum(childcounts),chisq(childcounts) > th))
                #newdkt[inst] = (
                #    np.array(childcounts,np.uint64),sum(childcounts),
                #    chisq(childcounts) > th,
                #    h(childcounts))
                newdkt_floats.append(h(childcounts))
        newdkts_ints.append(np.array(newdkt_ints,np.uint64))
        newdkts_floats.append(np.array(newdkt_floats,np.float64))
        #print('new',len(newdkt))
    #fullinsts = newdkts

    #print(newdkts_ints[0])
    #sys.exit()

    #for x in newdkts_ints:
    #    print(x.shape)
    
    return newdkts_ints, newdkts_floats
    

def save_local_scores(local_scores, filename):
    variables = local_scores.keys()
    with open(filename, "w") as scores_file:
        scores_file.write(str(len(variables)))
        for child, dkt in local_scores.items():
            scores_file.write("\n" + child + " " + str(len(dkt.keys())))
            for parents, score in dkt.items():
                #highest_sup = None
                scores_file.write("\n" + str(score) + " " + str(len(parents)) +" "+ " ".join(parents))

def prune_local_score(this_score, parent_set, child_dkt):
    for other_parent_set, other_parent_set_score in child_dkt.items():
        if other_parent_set_score >= this_score and other_parent_set < parent_set:
            return True
    return False

def fromdataframe(df):
    cols = []
    arities = []
    varnames = []
    for varname, vals in df.items():
        varnames.append(varname)
        cols.append(vals.cat.codes)
        arities.append(len(vals.cat.categories))
    return np.transpose(np.array(cols,dtype=np.uint32)), arities, varnames

class DiscreteData:
    """
    Complete discrete data
    """
    
    def __init__(self, data_source, varnames = None, arities = None):
        '''Initialises a `DiscreteData` object.

        If  `data_source` is a filename then it is assumed that:

            #. All values are separated by whitespace
            #. Comment lines start with a '#'
            #. The first line is a header line stating the names of the 
               variables
            #. The second line states the arities of the variables
            #. All other lines contain the actual data

        Args:
          data_source (str/array_like/Pandas.DataFrame) : 
            Either a filename containing the data or an array_like object or
            Pandas data frame containing it.

          varnames (iter) : 
           Variable names corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the variable names). Otherwise if not supplied (`=None`)
           then variables names will be: X1, X2, ...

          arities (iter) : 
           Arities for the variables corresponding to columns in the data.
           Ignored if `data_source` is a filename or Pandas DataFrame (since they 
           will supply the arities). Otherwise if not supplied (`=None`)
           the arity for each variable will be set to the number of distinct values
           observed for that variable in the data.
        '''

        if type(data_source) == str:
            with open(data_source, "r") as file:
                line = file.readline().rstrip()
                while line[0] == '#':
                    line = file.readline().rstrip()
                varnames = line.split()
                line = file.readline().rstrip()
                while line[0] == '#':
                    line = file.readline().rstrip()
                arities = np.array([int(x) for x in line.split()],dtype=np.uint32)

                for arity in arities:
                    if arity < 2:
                        raise ValueError("This line: '{0}' is interpreted as giving variable arities but the value {1} is less than 2.".format(line,arity))

                # class whose instances are callable functions 'with memory'
                class Convert:
                    def __init__(self):
                        self._last = 0 
                        self._dkt = {}

                    def __call__(self,s):
                        try:
                            return self._dkt[s]
                        except KeyError:
                            self._dkt[s] = self._last
                            self._last += 1
                            return self._dkt[s]


                converter_dkt = {}
                for i in range(len(varnames)):
                    # trick to create a function 'with memory'
                    converter_dkt[i] = Convert()
                data = np.loadtxt(file,
                                  dtype=np.uint32,
                                  converters=converter_dkt,
                                  comments='#')

        elif type(data_source) == pd.DataFrame:
            data, arities, varnames = fromdataframe(data_source)
        else:
            data = np.array(data_source,dtype=np.uint32)
        self._data = data
        self._data_length = data.shape[0]
        if arities is None:
            self._arities = np.array([x+1 for x in data.max(axis=0)],dtype=np.uint32)
        else:
            self._arities = np.array(arities,dtype=np.uint32)
        if varnames is None:
            self._variables = ['X{0}'.format(i) for i in range(1,len(self._arities)+1)]
        else:
            # order of varnames determined by header line in file, if file used
            self._variables = varnames
        self._varidx = {}
        for i, v in enumerate(self._variables):
            self._varidx[v] = i

        self._unique_data, counts = np.unique(self._data, axis=0, return_counts=True)
        self._unique_data_counts = np.array(counts,np.uint32)
            
        self._maxflatcontabsize = 1000000
        
    def data(self):
        '''
        The data with all values converted to unsigned integers.

        Returns:
         pandas.DataFrame: The data
        '''

        df = pd.DataFrame(self._data,columns=self._variables)
        arities = self._arities
        for i, (name, data) in enumerate(df.items()):
            # ensure correct categories are recorded even if not
            # all observed in data
            df[name] = pd.Categorical(data,categories=range(arities[i]))
        return df

    def rawdata(self):
        '''
        The data with all values converted to unsigned integers.
        And without any information about variable names or aritiies.

        Returns:
         numpy.ndarray: The data
        '''
        return self._data
    
    def data_length(self):
        '''
        Returns:
         int: The number of datapoints in the data
        '''
        return self._data_length

    def arities(self):
        '''
        Returns:
         numpy.ndarray: The arities of the variables.
        '''
        return self._arities

    def arity(self,v):
        '''
        Args:
         v (str) : A variable name
        Returns:
         int : The arity of `v`
        '''

        return self._arities[self._varidx[v]]

    def variables(self):
        '''
        Returns:
         list : The variable names
        '''
        return self._variables

    def varidx(self):
        '''
        Returns:
         dict : Maps a variable name to its position in the list of variable names.
        '''
        return self._varidx

    def contab(self,variables):
        cols = np.array([self._varidx[v] for v in variables], dtype=np.uint32)
        cols.sort() 
        return make_contab(self._unique_data,self._unique_data_counts,cols,self._arities[cols],self._maxflatcontabsize)[0]
    


class LLPenalised(DiscreteData):
    '''Abstract class for penalised log likelihood scores
    '''

    def __init__(self,data):
        '''Initialises a `LLPenalised` object.

        Args:
         data (DiscreteData): data
        '''
        self.__dict__.update(data.__dict__)
        self._maxllh = {}
        for i, v in enumerate(self._variables):
            self._maxllh[v] = self.ll_score(v,self._variables[:i]+self._variables[i+1:])[0]
            
    def ll_score(self,child,parents):
        '''
        The fitted log-likelihood score for `child` having `parents`

        In addition to the score the number of joint instantations of the parents is returned.
        If this number would cause an overflow `None` is returned instead of the number.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: (1) The fitted log-likelihood local score for the given family and 
                (2) the number of joint instantations of the parents (or None if too big)
        '''
        family_cols = np.array(sorted([self._varidx[x] for x in list(parents)+[child]]), dtype=np.uint32)
        contab, strides = make_contab(self._unique_data, self._unique_data_counts, family_cols,
                                      self._arities[family_cols], self._maxflatcontabsize)

        child_col = self._varidx[child]
        child_arity = self._arities[child_col]
        child_idx = tuple(family_cols).index(child_col) # where in family_cols is the col for the child
            
        if len(contab) > 0: #if successfully created standard flat contab
            return _compute_ll_from_flat_contab(contab,strides,child_idx,child_arity), len(contab)/child_arity
        else:
            #have to make contab represented by unique insts
            pa_cols = np.delete(family_cols,child_idx)
            pa_uniqs, pa_idxs = np.unique(data[:,pa_cols],axis=0,return_inverse=True) # numba cannot handle return_inverse arg
            return _compute_ll_from_unique_contab(self._unique_data, self._unique_data_counts,
                                                  len(pa_uniqs), pa_idxs, child_arity, child_col), None
    
    def score(self,child,parents):
        '''
        Return LL score minus complexity penalty for `child` having `parents`, 
        and also upper bound on the score for proper supersets.

        To compute the penalty the number of joint instantations is multiplied by the arity
        of the child minus one. This value is then multiplied by log(N)/2 for BIC and 1 for AIC.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Raises:
         ValueError: If the number of joint instantations of the parents would cause an overflow
          when computing the penalty
        
        Returns:
         tuple: The local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        this_ll_score, numinsts = self.ll_score(child,parents)
        if numinsts is None:
            raise ValueError('Too many joint instantiations of parents {0} to compute penalty'.format(parents))
        penalty = numinsts * self._child_penalties[child]
        # number of parent insts will at least double if any added
        return this_ll_score - penalty, self._maxllh[child] - (penalty*2)


class LL(LLPenalised):

    def __init__(self,data):
        '''Initialises a `LL` object.

        Args:
         data (DiscreteData): data
        '''
        super(LL,self).__init__(data)

    def score(self,child,parents):
        '''
        Return the fitted log-likelihood score for `child` having `parents`, 
        and also upper bound on the score for proper supersets of `parents`.

        Args:
         child (str): The child variable
         parents (iter): The parent variables

        Returns:
         tuple: The fitted log-likelihood local score for the given family and an upper bound on the local score 
         for proper supersets of `parents`
        '''
        return self.ll_score(child,parents)[0], self._maxllh[child]
        
class BIC(LLPenalised):
    """
    Discrete data with attributes and methods for BIC scoring
    """
    def __init__(self,data):
        '''Initialises a `BIC` object.

        Args:
         data (DiscreteData): data
        '''
        super(BIC,self).__init__(data)
        fn = 0.5 * log(self._data_length)  # Carvalho notation
        self._child_penalties = {v:fn*(self.arity(v)-1) for v in self._variables}

class AIC(LLPenalised):
    """
    Discrete data with attributes and methods for AIC scoring
    """
    def __init__(self,data):
        '''Initialises an `AIC` object.

        Args:
         data (DiscreteData): data
        '''
        super(AIC,self).__init__(data)
        self._child_penalties = {v:self.arity(v)-1 for v in self._variables}

class BDeu(DiscreteData):
    """
    Discrete data with attributes and methods for BDeu scoring
    """

    def __init__(self,data,alpha=1.0):
        '''Initialises a `BDeu` object.

        Args:
         data (DiscreteData): data
         
         alpha (float): The *equivalent sample size*
        '''
        self.__dict__.update(data.__dict__)
        self.alpha = alpha
        self._cache = {}

        # for upper bounds
        self._atoms = get_atoms(self._data,self._arities)

        
    @property
    def alpha(self):
        '''float: The *equivalent sample size* used for BDeu scoring'''
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        '''Set the *equivalent sample size* for BDeu scoring
        
        Args:
         alpha (float): the *equivalent sample size* for BDeu scoring

        Raises:
         ValueError: If `alpha` is not positive
        '''
        if not alpha > 0:
            raise ValueError('alpha (equivalent sample size) must be positive but was give {0}'.format(alpha))
        self._alpha = alpha

    def clear_cache():
        '''Empty the cache of stored BDeu component scores

        This should be called, for example, if new scores are being computed
        with a different alpha value
        '''
        self._cache = {}
        
    def upper_bound_james(self,child,parents,alpha=None):
        """
        Compute an upper bound on proper supersets of parents

        Args:
         child (str) : Child variable.
         parents (iter) : Parent variables
         alpha (float) : ESS value for BDeu score. If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : An upper bound on the local score for parent sets
         for `child` which are proper supersets of `parents`

        """
        if alpha is None:
            alpha = self._alpha
        child_idx = self._varidx[child]
        pa_idxs = sorted([self._varidx[v] for v in parents])
        for pa_idx in pa_idxs:
            alpha /= self._arities[pa_idx]
        r = self._arities[child_idx]

        # each element of atoms_ints is a tuple of ints:
        # (fullinst,childvalcounts,sum(childvalcounts),ok_first)
        # each element of atoms_floats is:
        # sum_n n*log(n/tot), where sum is over childvalcounts
        # and tot = sum(childvalcounts)
        atoms_ints, atoms_floats = self._atoms[0][child_idx], self._atoms[1][child_idx]

        if len(atoms_floats) == 0:
            return 0.0

        # remove cols corresponding to non-parents and order
        p = len(self._arities)
        end_idxs = list(range(p,p+r+2))
        atoms_ints_redux = atoms_ints[:,pa_idxs+end_idxs]
        if len(pa_idxs) > 0:
            idxs = np.lexsort([atoms_ints_redux[:,col] for col in range(len(pa_idxs))])
        else:
            idxs = list(range(len(atoms_floats)))

        return upper_bound_james_fun(atoms_ints_redux,atoms_floats,len(pa_idxs),alpha,r,idxs)
        
    def bdeu_score_component(self,variables,alpha=None):
        '''Compute the BDeu score component for a set of variables
        (from the current dataset).

        The BDeu score for a child v having parents Pa is 
        the BDeu score component for Pa subtracted from that for v+Pa
        
        Args:
         variables (iter) : The names of the variables
         alpha (float) : The effective sample size parameter for the BDeu score.
          If not supplied (=None)
          then the value of `self.alpha` is used.

        Returns:
         float : The BDeu score component.
        '''
        if alpha is None:
            alpha = self._alpha

        if len(variables) == 0:
            return lgamma(alpha) - lgamma(alpha + self._data_length), 1
        else:
            cols = np.array(sorted([self._varidx[x] for x in list(variables)]), dtype=np.uint32)
            return compute_bdeu_component(
                self._unique_data,self._unique_data_counts,cols,
                np.array([self._arities[i] for i in cols], dtype=np.uint32),
                alpha,self._maxflatcontabsize)


    def bdeu_score(self, child, parents):

        parents_set = frozenset(parents)
        try:
            parent_score, _ = self._cache[parents_set]
        except KeyError:
            parent_score, non_zero_count = self.bdeu_score_component(parents)
            self._cache[parents_set] = parent_score, non_zero_count

        family_set = frozenset((child,)+parents)
        try:
            family_score, non_zero_count = self._cache[family_set]
        except KeyError:
            family_score, non_zero_count = self.bdeu_score_component((child,)+parents)
            self._cache[family_set] = family_score, non_zero_count

        simple_ub = -log(self.arity(child)) * non_zero_count

        #james_ub = self.upper_bound_james(child,parents)
        
        #return parent_score - family_score, min(simple_ub,james_ub)
        return parent_score - family_score, simple_ub

        
    def bdeu_scores(self,palim=None,pruning=True,alpha=None):
        """
        Exhaustively compute all BDeu scores for all child variables and all parent sets up to size `palim`.
        If `pruning` delete those parent sets which have a subset with a better score.
        Return a dictionary dkt where dkt[child][parents] = bdeu_score
        
        Args:
         palim (int/None) : Limit on parent set size
         pruning (bool) : Whether to prune
         alpha (float) : ESS for BDeu score. 
          If not supplied (=None)
          then the value of `self.alpha` is used.



        Returns:
         dict : dkt where dkt[child][parents] = bdeu_score
        """
        if alpha is None:
            alpha = self._alpha
        
        if palim == None:
            palim = self._arities.size - 1

        score_dict = {}
        # Initialisation
        # Need to create dict for every child
        # also its better to do the zero size parent set calc here
        # so that we don't have to do a check for every parent set
        # to make sure it is not of size 0 when calculating score component size
        no_parents_score_component = lgamma(alpha) - lgamma(alpha + self._data_length)
        for c, child in enumerate(self._variables):
            score_dict[child] = {
                frozenset([]):
                no_parents_score_component
            }
        
        for pasize in range(1,palim+1):
            for family in combinations(self._variables,pasize): 
                score_component = self.bdeu_score_component(family,alpha)
                family_set = frozenset(family)
                for child in self._variables:
                    if child in family_set:
                        parent_set = family_set.difference([child])
                        score_dict[child][parent_set] -= score_component
                        if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                            del score_dict[child][parent_set]
                    else:
                        score_dict[child][family_set] = score_component 
                
        # seperate loop for maximally sized parent sets
        for vars in combinations(self._variables,palim+1):
            score_component = self.bdeu_score_component(vars,alpha)
            vars_set = frozenset(vars)
            for child in vars:
                parent_set = vars_set.difference([child])
                score_dict[child][parent_set] -= score_component
                if pruning and prune_local_score(score_dict[child][parent_set],parent_set,score_dict[child]):
                    del score_dict[child][parent_set]

            
        return score_dict

    
    

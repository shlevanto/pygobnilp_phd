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
   Python version of GOBNILP
"""

__author__ = "James Cussens"
__email__ = "james.cussens@york.ac.uk"

import argparse
from pygobnilp.gobnilp import Gobnilp
    
parser = argparse.ArgumentParser(description='Use Gurobi for Bayesian network learning',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('input', help="File containing input, either containing discrete data (default), continuous data (see --bge option) or local scores in 'Jaakkola' format (see --scores option)")
parser.add_argument('-n', '--nsols', type=int, default=1, help = "How many BNs to find")
parser.add_argument('-e', '--edge_penalty', type=float, default=0.0, help = "Edge penalty")
parser.add_argument('-k', '--kbest', action='store_true', help = "Whether to find 'k-best' networks")
parser.add_argument('-m', '--mec', action='store_true', help = "Whether to only allow one DAG per Markov Equivalence Class")
#parser.add_argument('-c', '--chordal', action='store_true', help = "Whether to only allow DAGs equivalent to chordal graphs")
parser.add_argument('-s','--scores', action="store_true", help="The input consists of pre-computed local scores (not discrete data)")
parser.add_argument('--bge', action="store_true", help="The input consists of continuous, not discrete, data and BGe scoring will be used")
parser.add_argument('--bic', action="store_true", help="The input consists of discrete data and BIC scoring should be used")
parser.add_argument('--aic', action="store_true", help="The input consists of discrete data and AIC scoring should be used")
parser.add_argument('--ll', action="store_true", help="The input consists of discrete data and LL scoring should be used")
parser.add_argument('-v', '--verbose', action="count", default=0, help="How verbose to be")
parser.add_argument('-g', '--gurobi_output', action="store_true", help="Whether to show Gurobi output")
parser.add_argument('-o', '--output_file', help="PDF (or DOT) file for learned BN")
parser.add_argument('--consfile', help="Python module defining user constraints")
parser.add_argument('--params', help="Gurobi parameter settings")
parser.add_argument('--starts', help="Starting DAG(s) in bnlearn modelstring format")

# Scores generation options
parser.add_argument('-u','--nopruning', action="store_true", help="No pruning of local scores to be done")
parser.add_argument('-p','--palim', type=int, default=3, help="Parent set size limit (for local score generation)")
parser.add_argument('-a', '--alpha', type=float, default=1.0, help="The equivalent sample size (for BDeu local score generation)")
#parser.add_argument('--nu', type=float, default=None, help ="the mean vector for the Normal part of the normal-Wishart prior for BGe scoring. If None then the sample mean is used.")
parser.add_argument('--alpha_mu', type=float, default=1.0, help = "imaginary sample size value for the Normal part of the normal-Wishart prior for BGe scoring")
parser.add_argument('--alpha_omega', type=int, default=None, help = "Degrees of freedom for the Wishart part of the normal-Wishart prior for BGe scoring. Must be at least the number of variables. If None then set to 2 more than the number of variables.")

args = parser.parse_args()

model = Gobnilp(verbose=args.verbose,gurobi_output=args.gurobi_output,params=args.params)

# arguments common to all types of learning
argsdict = {}
for arg in 'consfile', 'nsols', 'kbest', 'mec', 'palim', 'output_file':
    argsdict[arg] = getattr(args,arg)

if args.scores:
    model.learn(local_scores_source=args.input,**argsdict)
else:
    # arguments common to learning from data
    argsdict['data_source'] = args.input
    argsdict['edge_penalty'] = args.edge_penalty
    argsdict['pruning'] = not args.nopruning
    if args.starts is not None:
        argsdict['starts'] = args.starts.split()
    if args.bge:
        model.learn(data_type='continuous',local_score_type='BGe',
                    alpha_mu=args.alpha_mu, alpha_omega=args.alpha_omega,
                    **argsdict)
    else:
        if args.bic:
            lst='BIC'
        elif args.aic:
            lst='AIC'
        elif args.ll:
            lst='LL'
        else:
            lst='BDeu'
        model.learn(data_type='discrete',local_score_type=lst,
                    alpha=args.alpha,**argsdict)

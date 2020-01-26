#!/usr/bin/env python
#/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *   GOBNILP (Python version) Copyright (C) 2020 James Cussens           *
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
   Command line script for Python version of GOBNILP
"""

__author__ = "James Cussens"
__email__ = "james.cussens@york.ac.uk"

import argparse
from pygobnilp.gobnilp import Gobnilp
    
parser = argparse.ArgumentParser(description='Use Gurobi for Bayesian network learning',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_source", help="File containing data or local scores")
parser.add_argument("--header", action="store_true",default=True,
                    help="For continuous data only: A header containing variable names is the first non-comment line in the file.")
parser.add_argument("--comments", default='#',
                    help="For continuous data only: Lines starting with this string are treated as comments.")
parser.add_argument("--delimiter", action="store_true",default=None,
                    help="For continuous data only: String used to separate values. If not set then whitespace is used.")
parser.add_argument("--end", default="output written",
                    help="End stage for learning. If set to 'local scores' execution stops once local scores are computed")
parser.add_argument("--score", default="BDeu",
                    help="""Name of scoring function used for computing local scores. Must be one
                    of the following: BDeu, BGe, DiscreteLL,
                    DiscreteBIC, DiscreteAIC, GaussianLL, GaussianBIC,
                    GaussianAIC, GaussianL0.""")
parser.add_argument("--k", default=1,
                    help="""Penalty multiplier for penalised log-likelihood scores (eg BIC, AIC) or tuning parameter ('lambda^2) for l_0
                    penalised Gaussian scoring (as per van de Geer and Buehlmann)""")
parser.add_argument("--ls", action="store_true",
                    help="For Gaussian scores, make unpenalised score should -(1/2) * MSE, rather than log-likelihood")
parser.add_argument("--standardise", action="store_true",
                    help="Standardise continuous data.")
parser.add_argument("-p", "--palim", type=int, default=3,
                    help="Maximum size of parent sets.")
parser.add_argument("--alpha", type=float, default=1.0,
                    help="The equivalent sample size for BDeu local score generation.")
parser.add_argument("--alpha_mu", type=float, default=1.0,
                    help="Imaginary sample size value for the Normal part of the normal-Wishart prior for BGe scoring.")
parser.add_argument("--alpha_omega", type=int, default=None,
                    help="""Degrees of freedom for the Wishart part of the normal-Wishart prior for BGe scoring. 
                    Must be at least the number of variables. If not supplied 2 more than the number of variables is used.""")
parser.add_argument('-s','--scores', action="store_true", help="The input consists of pre-computed local scores (not data)")
parser.add_argument("-n", "--nsols", type=int, default=1, help="Number of BNs to learn")
parser.add_argument("--kbest", action="store_true",
                    help="Whether the nsols learned BNs should be a highest scoring set of nsols BNs.")
parser.add_argument("--mec", action="store_true",
                    help="Make only one BN per Markov equivalence class feasible.")
parser.add_argument("--consfile",
                    help="A file (Python module) containing user constraints.")
parser.add_argument("--settingsfile",
                    help="""A file (Python module) containing values for the arguments for Gobnilp's 'learn' method
                    Any such values override both default values and any values set on the command line.""")
parser.add_argument("--nopruning", action="store_true",
                    help="No pruning of provably sub-optimal parent sets.")
parser.add_argument("--edge_penalty", type=float, default=0.0,
                    help="The local score for a parent set with p parents will be reduced by p*edge_penalty.")
parser.add_argument("--noplot", action="store_true",
                    help="Prevent learned BNs/CPDAGs being plotted.")
parser.add_argument("--noabbrev", action="store_true",
                    help="When plotting DO NOT to abbreviate variable names to the first 3 characters.")
parser.add_argument("--output_scores",
                    help="Name of a file to write local scores")
parser.add_argument("-o", "--output_stem",
                    help="""Learned BNs will be written to 'output_stem.ext' for each extension defined by 
                    `output_ext`. If multiple DAGs have been learned then output files are called 'output_stem_0.ext',
                    'output_stem_1.ext' ... No DAGs are written if this is not set.""")
parser.add_argument("--nooutput_dag", action="store_true",
                    help="Do not write DAGs to any output files")
parser.add_argument("--nooutput_cpdag", action="store_true",
                    help="Do not write CPDAGs to any output files")
parser.add_argument("--output_ext",default='pdf',
                    help="Comma separated file extensions which determine the format of any output DAGs or CPDAGs.")
parser.add_argument("-v", "--verbose", type=int, default=0,
                    help="How much information to show when adding variables and constraints (and computing scores)")
parser.add_argument("-g", "--gurobi_output", action="store_true", help="Whether to show output generated by Gurobi.")
                    
args = parser.parse_args()
argdkt = vars(args)

# now alter argument dictionary to send to learn method

# process options which set 'learn' arguments to false
for opt in 'nopruning', 'noplot', 'noabbrev', 'nooutput_dag', 'nooutput_cpdag':
    argdkt[opt[2:]] = not argdkt[opt]
    del argdkt[opt]
    
# interpret first argument as local scores file if --scores is used
if argdkt['scores']:
    argdkt['local_scores_source'] = argdkt['data_source']
    del argdkt['data_source']
del argdkt['scores']

#convert string specifying list of extensions to a list
argdkt['output_ext'] = argdkt['output_ext'].split(',')

#assume data is continuous if a score for continuous data is specified
s = argdkt['score']
if s == "BGe" or s.startswith('Gaussian'):
    argdkt['data_type'] = 'continuous'

# do learning
Gobnilp().learn(**argdkt)

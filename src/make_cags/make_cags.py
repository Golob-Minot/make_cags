#!/usr/bin/env python3

import anndata as ad
import scipy
import numpy as np
import pandas as pd
import pynndescent
import logging
import time
import argparse
import warnings
warnings.filterwarnings("ignore")


def make_cags(
        gene_ad,
        MAX_ITERATIONS=100,
        DISTANCE_THRESHOLD=0.1,
        N_NEIGHBORS=10
):
    # ## Strategy
    # 
    # Initially:
    # 1. Make each gene its own CAG
    # 2. Seed the currentn abundance matrix
    # 
    # Iteratively:
    # 1. Find the pairwise distance of the k-nearest neighbors for each CAG
    # 2. Combine together CAGS below the current theshold distance
    # 3. Recalculate the current abundance matrix
    # 

    gene_ad.var['CAG'] = [
        f'CAG_{i+1:08d}' 
        for i in range(len(gene_ad.var_names))
    ]
    
    cur_cags = list(gene_ad.var['CAG'])
    cur_cag_abd_matrix = gene_ad.X.T

    for itr in range(MAX_ITERATIONS):    
        # Each iteration
        # Get the pairwise distance...
        logging.info(f"Building ANN Index for iteration {itr+1}")
        # Build this iteration's nn index
        itr_nnindex = pynndescent.NNDescent(
            cur_cag_abd_matrix,
            metric="cosine",
            random_state=42,
            n_neighbors=N_NEIGHBORS,        
        )
        # And use it to find the nearest neighbors for the current CAGs
        logging.info(f'Finding {N_NEIGHBORS} nearest neighbors for each CAG')
        itr_nn = itr_nnindex.query(
            cur_cag_abd_matrix
        )
        logging.info("Converting to long format")
        # Convert to long format
        pwd_l = pd.DataFrame(
            [
                (
                    cur_cags[i], # index of CAG I
                    cur_cags[itr_nn[0][i][j]], # Index of CAG J 
                    itr_nn[1][i][j], # Distance
                )
                for i in range(len(itr_nn[0]))
                for j in range(N_NEIGHBORS)
            ],
            columns=['I', 'J', 'Distance'],
        )
        # And filter it down to remove self-to-self and distances above threshold..
        logging.info("Filtering down to combinable CAGs")
        
        pwd_l = pwd_l[
            (pwd_l.I < pwd_l.J) &
            (pwd_l.Distance <= DISTANCE_THRESHOLD)
        ]    
        
        logging.info(f'{len(pwd_l)} CAGs to be combined')
        
        if len(pwd_l) == 0:
            logging.info("COMPLETE!")
            break

        logging.info("Regrouping genes")    
        # Regroup genes
        gene_ad.var['CAG'].replace({
            r.J: r.I
            for i, r in 
            pwd_l[['I', 'J']].groupby("I").value_counts().reset_index().iterrows()
        }, inplace=True)
        logging.info(f'There are {gene_ad.var.CAG.nunique()} CAGS')
        logging.info("Rebuilding CAG abundance matrix")
        # Rebuild CAG abd matrix
        cur_cags = list(gene_ad.var.CAG.unique())
        # Make an empty sparse LoL matrix
        cur_cag_abd_matrix = scipy.sparse.lil_matrix(
            (
                len(cur_cags),
                len(gene_ad.obs)
            ),
            dtype=np.int32
        )
        for cag_i, cag in enumerate(cur_cags):
            cur_cag_abd_matrix[cag_i] = gene_ad.X[
                :, gene_ad.var.CAG == cag
            ].sum(axis=1)
        cur_cag_abd_matrix = cur_cag_abd_matrix.tocsr()
        logging.info("Completed building matrix")    


    logging.info("Making CAG AnnData")
    # Make a final CAG Ad with abundance
    CAG_ad = ad.AnnData(
        cur_cag_abd_matrix.T.tocsr(),
        obs=gene_ad.obs,
        var=pd.DataFrame(index=cur_cags)
    )
    CAG_ad.layers['fract'] = (CAG_ad.X / CAG_ad.X.sum(axis=1)).tocsr()
    CAG_ad.layers['RPM'] = (CAG_ad.layers['fract'] * 1e6).astype(np.int32)

    # Coverage (i.e. what proportion of genes in this CAG are found in each specimen)
    cur_cags = list(CAG_ad.var_names)
    cur_cag_coverage_matrix = scipy.sparse.lil_matrix(
        (
            len(CAG_ad.var),
            len(CAG_ad.obs)
        ),
        dtype=np.int32
    )

    for cag_i, cag in enumerate(cur_cags):
        cur_cag_coverage_matrix[cag_i] = (gene_ad.X[
            :, gene_ad.var.CAG == cag
        ] > 0).sum(axis=1)
    CAG_ad.layers['coverage_n'] = cur_cag_coverage_matrix.T.tocsr()
    # How many genes in a CAG?
    CAG_ad.var['total_genes'] = gene_ad.var.CAG.value_counts().loc[
        CAG_ad.var_names
    ]
    # And proportion of a CAG's genes detectable in this specimen...
    CAG_ad.layers['coverage_p'] = (CAG_ad.layers['coverage_n'] / gene_ad.var.CAG.value_counts().loc[
        CAG_ad.var_names
    ].values).tocsr()
    
    return CAG_ad


def main():
    """Make CAGS from a gene abundance anndata"""
    parser = argparse.ArgumentParser(
        description="""Make CAGS (co-abundant genes) from a gene abundance anndata""")

    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Location for gene abundance data in an H5AD 'anndata' format")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="Location for CAG abundance data in H5AD 'anndata' format")
    parser.add_argument("--output-genes",
                        type=str,
                        help="(Optional) Where to output the modified gene-abundance H5AD now with CAG membership in var")
    parser.add_argument("--logfile",
                        type=str,
                        help="""(Optional) Write log to this file.""")
    parser.add_argument("--threshold",
                        default=0.1,
                        type=float,
                        help="""Threshold cosine distance for making CAGs. Default 0.1""")
    parser.add_argument("--max-iterations",
                        default=1000,
                        type=int,
                        help="""Maximum number of iterations. Default 1000""")
    parser.add_argument("--n-neighbors",
                        default=10,
                        type=int,
                        help="""Nearest neighbors to find. Default 10.""")

    args = parser.parse_args()

    start_time = time.time()

    # Set up logging
    logFormatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [make_cags] %(message)s'
    )
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    if args.logfile:
        # Write to file
        fileHandler = logging.FileHandler(args.logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    # Write to STDOUT
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    
    logging.info("Attempting to load gene abundance H5AD")
    
    gene_ad = ad.read_h5ad(args.input)
    
    logging.info("Starting to make CAGs")
    CAG_ad = make_cags(
        gene_ad,
        MAX_ITERATIONS=args.max_iterations,
        N_NEIGHBORS=args.n_neighbors,
        DISTANCE_THRESHOLD=args.threshold,
    )

    logging.info("Outputting CAG abundance")
    CAG_ad.write_h5ad(
        args.output
    )

    if args.output_genes:
        gene_ad.write_h5ad(
            args.output_genes
        )

    elapsed = round(time.time() - start_time, 2)
    logging.info("Time elapsed: {:,} seconds".format(elapsed))


if __name__ == "__main__":
    main()

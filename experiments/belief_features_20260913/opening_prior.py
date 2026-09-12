"""Analytic one-piece marginals of the randomized setup generator at move zero.

The generator chooses a flag, then mines, then bombs, then uniformly shuffles
the remaining ranked pieces. Each constrained stage has a fixed pool size.
This is a distributional opening reference, not a whole-game Bayes error bound.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from junqi_core.info_model import TRACKED_TYPES
from junqi_core.rules import (PieceType,PIECE_COUNTS,CAMP_INDICES,STRONGHOLD_INDICES,
                              BACK_TWO_ROWS_INDICES,FRONT_ROW_INDICES)
from experiments.belief_features_20260913.diagnostics import load,score_predictions,tabular_probabilities


def exact_marginal_table():
    slots=[i for i in range(30) if i not in CAMP_INDICES]
    back=sum(s in BACK_TWO_ROWS_INDICES for s in slots)
    nonfront=sum(s not in FRONT_ROW_INDICES for s in slots)
    mines=PIECE_COUNTS[PieceType.DILEI]; bombs=PIECE_COUNTS[PieceType.ZHADAN]
    assert PIECE_COUNTS[PieceType.JUNQI]==1
    special=(PieceType.JUNQI,PieceType.DILEI,PieceType.ZHADAN)
    ranked=[t for t in TRACKED_TYPES if t not in special]
    ranked_total=sum(PIECE_COUNTS[t] for t in ranked)
    table=np.zeros((len(slots),12))
    for row,slot in enumerate(slots):
        flag=1/len(STRONGHOLD_INDICES) if slot in STRONGHOLD_INDICES else 0.
        mine=(1-flag)*mines/(back-1) if slot in BACK_TWO_ROWS_INDICES else 0.
        bomb=(1-flag-mine)*bombs/(nonfront-1-mines) if slot not in FRONT_ROW_INDICES else 0.
        for kind,value in zip(special,(flag,mine,bomb),strict=True): table[row,TRACKED_TYPES.index(kind)]=value
        for kind in ranked: table[row,TRACKED_TYPES.index(kind)]=(1-flag-mine-bomb)*PIECE_COUNTS[kind]/ranked_total
    np.testing.assert_allclose(table.sum(1),1,atol=1e-12)
    np.testing.assert_allclose(table.sum(0),[PIECE_COUNTS[t] for t in TRACKED_TYPES],atol=1e-12)
    return slots,table


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    torch.set_num_threads(1)
    slots,table=exact_marginal_table()
    result={'reference':'analytic marginals of the uniform randomized setup procedure at move zero',
        'uses_training_or_test_labels_to_fit':False,'slots':slots,'piece_types':[t.name for t in TRACKED_TYPES],
        'probabilities':table.tolist(),'mean_opening_marginal_entropy_nats':float(-(table*np.log(table.clip(1e-30))).sum()/len(slots)),
        'mean_opening_marginal_top1':float(table.max(1).mean()),
        'scope':'Assumes the documented randomized lineup distribution. The entropy is an opening distribution reference, not a finite-dataset or whole-game irreducible-error estimate.',
        'evaluations':{}}
    for split in ('validation','test','ood_attack','ood_random'):
        raw=load(args.data/(split+'.pt.gz')); chosen=raw['step']==0
        data={k:v[chosen] for k,v in raw.items() if isinstance(v,torch.Tensor)}
        del raw
        probs=tabular_probabilities(data,torch.from_numpy(table).float())
        scores=score_predictions(probs,data)
        result['evaluations'][split]=scores['all']
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('mean_opening_marginal_entropy_nats','mean_opening_marginal_top1','scope')}),flush=True)


if __name__=='__main__': main()

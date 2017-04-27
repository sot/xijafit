import pytest
import sys
from pathlib import Path # if you haven't already done so
root = str(Path(__file__).resolve().parents[0])
sys.path.append(root)

import xijafit

BASE_MODEL = root + '/tcylaft6_model_spec_roll_base.json'

def test_fit():
    model = xijafit.XijaFit(BASE_MODEL, set_data_exprs=(u'tcylaft6_0=22.0',), start='2015:300',
      stop='2016:300',quiet=False, name='tcylaft6')

    model.freeze_all()
    model.thaw_solarheat_p()
    model.thaw_solarheat_roll()
    model.thaw_param(u'coupling__tcylaft6__tcylaft6_0__tau')
    model.thaw_param(u'solarheat__tcylaft6_0__ampl')
    model.fit(method='simplex')

    assert model.snapshots[0]['fit_stat'] > model.snapshots[1]['fit_stat']

if __name__ == "__main__":
    test_fit()
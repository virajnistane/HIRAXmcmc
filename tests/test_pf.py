from hiraxmcmc.util.cosmoparamvalues import ParametersFixed

def test_pf():
     pf = ParametersFixed()
     assert pf.cosmoparams_fixed == {'H0': 67.8, 'Omk': 0.0, 'Oml': 0.684, 'w0': -1.0, 'wa': 0.0}

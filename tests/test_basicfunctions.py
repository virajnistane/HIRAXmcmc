from hiraxmcmc.util import basicfunctions

def test_freq2z():
    assert round(basicfunctions.freq2z(500),2) == 1.86 

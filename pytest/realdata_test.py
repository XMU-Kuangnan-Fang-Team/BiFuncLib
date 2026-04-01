from BiFuncLib.real_data import tcell, growth

def test_realdata():
    tcell_data = tcell()['data']
    tcell_label = tcell()['label']
    growth_data = growth()['data']
    growth_label = growth()['label']
    x = growth()['location']
    assert tcell_data is not None
    assert tcell_label is not None
    assert growth_data is not None
    assert growth_label is not None
    assert x is not None
    


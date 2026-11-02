from touhou_rl.tactical_uncertain_witness import LinearHazard,witness_over_hypotheses


def test_uncertified_enumeration_cannot_become_safe_label():
    h=[LinearHazard(100,100,0,0,1)]
    r=witness_over_hypotheses([(0,0),(1,0)],h,[{'time_shift':0,'complete_pool':True}])
    assert r['status']=='unknown' and not r['positive']


def test_endpoint_safe_but_between_frame_crossing_rejected():
    h=[LinearHazard(5,-5,0,10,0)]
    r=witness_over_hypotheses([(0,0),(10,0)],h,[{'time_shift':0,'complete_pool':True}],player_radius=.1,coverage_certified=True)
    assert r['status']=='rejected_witness'
    assert r['results'][0]['collision']['relative_fraction']==.5


def test_one_alignment_collides_blocks_positive_set():
    h=[LinearHazard(3,0,-2,0,0)]
    hypotheses=[{'time_shift':-1,'complete_pool':True},{'time_shift':1,'complete_pool':True}]
    r=witness_over_hypotheses([(0,0),(0,0)],h,hypotheses,player_radius=1,coverage_certified=True)
    assert not r['positive']


def test_incomplete_pool_unknown_and_conditional_scope():
    hazard=[LinearHazard(100,100,0,0,1)]
    assert witness_over_hypotheses([(0,0),(1,0)],hazard,[{'time_shift':0,'complete_pool':False}],coverage_certified=True)['status']=='unknown'
    result=witness_over_hypotheses([(0,0),(1,0)],hazard,[{'time_shift':0,'complete_pool':True}],coverage_certified=True)
    assert result['positive'] and result['status']=='conditional_positive'

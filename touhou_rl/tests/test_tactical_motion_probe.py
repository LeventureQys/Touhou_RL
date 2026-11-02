from tools.rl.v373_motion_probe import bracket


def test_bracket_not_exact_without_nonmatch():
    assert bracket([{'gameplay_frame':5,'gui_input_current':129}],129)==[None,5]


def test_bracket_records_nonmatch_to_match():
    rows=[{'gameplay_frame':5,'gui_input_current':129},
          {'gameplay_frame':6,'gui_input_current':1}]
    assert bracket(rows,1)==[5,6]
    assert bracket(rows,65) is None

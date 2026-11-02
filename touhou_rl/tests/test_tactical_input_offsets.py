"""Actual globals have four-byte spacing despite u16 payloads."""
import ctypes
import struct
from touhou_rl import spec
from touhou_rl.state import GameStateReader


class ZeroMemory:
    module_base=spec.IMAGE_BASE
    def read_into(self,address,buffer):
        data=bytearray(ctypes.sizeof(buffer))
        if address==spec.ADDR_LAST_FRAME_INPUT:
            struct.pack_into('<HHHH',data,0,65,0xeeee,129,0xdddd)
        if address==spec.ADDR_GUI_INPUT_CURRENT:
            struct.pack_into('<HHHH',data,0,133,0xcccc,69,0xbbbb)
        ctypes.memmove(buffer,bytes(data),len(data))
    def read(self,address,size):
        return bytes(size)


def test_reader_does_not_decode_padding_as_current_or_previous():
    state=GameStateReader(ZeroMemory()).read_state(with_bullets=False,with_title=False)
    assert state.last_input_mask==65
    assert state.input_mask==129
    assert state.gui_input_current==133
    assert state.gui_input_previous==69

"""One chip shown as the four-card mesh, for the qualification harnesses that run on a single card.

The four-card port's builders (the GDN state copies, the conv windows, the commit DMA, the attention fold and mask
programs, the ordered K/V write) loop over tp_shapes.chip_count() chips and build one program per chip. A qualification
window on ONE card (1x1 mesh) cannot open a (1, 4) mesh, but every per-chip program is the same program over that chip's
own buffers, so the card harnesses present the one chip to the builders as the chips they expect: get_device_tensors gives
the one shard `chips` times, and a MeshProgramDescriptor is a ViewProgram whose chip-0 program alone is launched (the other
chips' programs, built from the same shard addresses, are dropped and counted as phantom). This is the ChipView the
extent-reader harness's TwoChipView (optimisation/ttnn-op/k64j/extent_reader_card_b.py, chips=2) generalises: the same
contract, one class.

WHAT THIS CANNOT SEE (as TwoChipView's docstring says of chip 1): a per-chip indexing error - chip 3 given chip 0's buffer,
or chip 3's own buffer never written - is invisible here, because the phantom chips' programs never run. It is covered
only by the four-card window (HW-B: the audits on chips 0-3).

Nothing here imports ttnn: the real module is passed in.
"""

from contextlib import contextmanager
import sys


class ViewProgram:
    """A MeshProgramDescriptor as the builders make it: one ProgramDescriptor per chip of the view, at mesh coordinate
    (0, chip). realise() makes the real one-chip descriptor from chip 0's, once, and reuses it after."""

    def __init__(self, view):
        self.view, self.entries, self.realised = view, {}, None

    def __setitem__(self, key, program):
        if self.realised is not None:
            raise RuntimeError('ChipView: a launched program was changed')
        first, last = key
        if first != last or first[0] != 0 or not 0 <= first[1] < self.view.chips:
            raise ValueError('ChipView: one chip coordinate (0, 0..%d) per entry, got %r' % (self.view.chips - 1, key))
        if first[1] in self.entries:
            raise ValueError('ChipView: chip %d programmed twice' % first[1])
        self.entries[first[1]] = program

    def realise(self):
        if self.realised is None:
            if sorted(self.entries) != list(range(self.view.chips)):
                raise ValueError('ChipView: a program for every chip of the view required, got chips %r'
                                 % sorted(self.entries))
            ttnn = self.view.ttnn
            real = ttnn.MeshProgramDescriptor()
            coordinate = ttnn.MeshCoordinate(0, 0)
            real[ttnn.MeshCoordinateRange(coordinate, coordinate)] = self.entries[0]
            self.realised = real
            self.view.realised += 1
            self.view.phantom += len(self.entries) - 1
        return self.realised


class ChipView:
    """ttnn for the builders, on one chip shown as `chips`. Everything but the four names below is ttnn's own.
    installed() makes it sys.modules['ttnn'] for the modules' function-local imports and restores what was there."""

    def __init__(self, ttnn, chips=4):
        self.__dict__.update(ttnn=ttnn, chips=chips, realised=0, phantom=0, launches=0)

    def __getattr__(self, name):
        return getattr(self.__dict__['ttnn'], name)

    def get_device_tensors(self, tensor):
        shards = list(self.ttnn.get_device_tensors(tensor))
        if len(shards) != 1:
            raise RuntimeError('ChipView presents ONE chip as %d; this tensor has %d shards' % (self.chips, len(shards)))
        return shards * self.chips

    @staticmethod
    def MeshCoordinate(row, column):  # noqa: N802 - mirrors ttnn
        return (row, column)

    @staticmethod
    def MeshCoordinateRange(first, last):  # noqa: N802 - mirrors ttnn
        return (tuple(first), tuple(last))

    def MeshProgramDescriptor(self):  # noqa: N802 - mirrors ttnn
        return ViewProgram(self)

    def generic_op(self, tensors, program):
        if isinstance(program, ViewProgram):
            program = program.realise()
        self.launches += 1
        return self.ttnn.generic_op(tensors, program)

    @contextmanager
    def installed(self):
        saved = sys.modules.get('ttnn')
        sys.modules['ttnn'] = self
        try:
            yield self
        finally:
            if saved is None:
                sys.modules.pop('ttnn', None)
            else:
                sys.modules['ttnn'] = saved

import numpy as np
import pytest
from pulse.signals import pos, chrom, bandpass, spectral_metrics, roi_rank

def rgb_for(hr, fps=30, seconds=20):
    t=np.arange(int(fps*seconds))/fps; pulse=np.sin(2*np.pi*hr*t)
    return np.column_stack([120+2*pulse,100+.55*pulse,80-.3*pulse])

@pytest.mark.parametrize("hz,bpm",[(1,60),(1.333333,80),(1.5,90)])
def test_known_frequency(hz,bpm):
    fps=30; t=np.arange(int(fps*20))/fps
    # Known-frequency synthetic pulse waveform, passed through the production filter/PSD estimator.
    wave=bandpass(np.sin(2*np.pi*hz*t),fps); m=spectral_metrics(wave,fps)
    assert abs(m["heart_rate_bpm"]-bpm)<4

def test_chrom_is_independent():
    x=rgb_for(1.2); assert np.isfinite(pos(x,30)[1:-1]).all(); assert not np.allclose(pos(x,30),chrom(x))

def test_filter_nyquist_validation():
    with pytest.raises(ValueError): bandpass(np.ones(100),8,low=.7,high=4)

def test_roi_quality_threshold():
    rows=[{"roi":"cheek","quality":.7},{"roi":"forehead","quality":.2}]
    assert roi_rank(rows)["roi"]=="cheek"
    assert roi_rank(rows,.8) is None

def test_short_signal_is_unavailable():
    assert spectral_metrics(np.ones(10),30)["heart_rate_bpm"] is None

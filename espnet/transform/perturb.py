"""Pertub module."""

import librosa
import numpy
import scipy
import soundfile

from espnet.utils.io_utils import SoundHDF5File


class SpeedPerturbation(object):
    """SpeedPerturbation class.

    The speed perturbation in kaldi uses sox-speed instead of sox-tempo,
    and sox-speed just to resample the input,
    i.e pitch and tempo are changed both.

    "Why use speed option instead of tempo -s in SoX for speed perturbation"
    https://groups.google.com/forum/#!topic/kaldi-help/8OOG7eE4sZ8

    Warning:
        This function is very slow because of resampling.
        I recommmend to apply speed-perturb outside the training using sox.

    """

    def __init__(
        self,
        lower=0.9,
        upper=1.1,
        utt2ratio=None,
        keep_length=True,
        res_type="kaiser_best",
        seed=None,
    ):
        """Initialize class."""
        self.res_type = res_type
        self.keep_length = keep_length
        self.state = numpy.random.RandomState(seed)

        if utt2ratio is not None:
            self.utt2ratio = {}
            # Use the scheduled ratio for each utterances
            self.utt2ratio_file = utt2ratio
            self.lower = None
            self.upper = None
            self.accept_uttid = True

            with open(utt2ratio, "r") as f:
                for line in f:
                    utt, ratio = line.rstrip().split(None, 1)
                    ratio = float(ratio)
                    self.utt2ratio[utt] = ratio
        else:
            self.utt2ratio = None
            # The ratio is given on runtime randomly
            self.lower = lower
            self.upper = upper

    def __repr__(self):
        """Return string with details of class."""
        if self.utt2ratio is None:
            return "{}(lower={}, upper={}, " "keep_length={}, res_type={})".format(
                self.__class__.__name__,
                self.lower,
                self.upper,
                self.keep_length,
                self.res_type,
            )
        else:
            return "{}({}, res_type={})".format(
                self.__class__.__name__, self.utt2ratio_file, self.res_type
            )

    def __call__(self, x, uttid=None, train=True):
        """Process call function."""
        if not train:
            return x

        x = x.astype(numpy.float32)
        if self.accept_uttid:
            ratio = self.utt2ratio[uttid]
        else:
            ratio = self.state.uniform(self.lower, self.upper)

        # Note1: resample requires the sampling-rate of input and output,
        #        but actually only the ratio is used.
        y = librosa.resample(x, ratio, 1, res_type=self.res_type)

        if self.keep_length:
            diff = abs(len(x) - len(y))
            if len(y) > len(x):
                # Truncate noise
                y = y[diff // 2 : -((diff + 1) // 2)]
            elif len(y) < len(x):
                # Assume the time-axis is the first: (Time, Channel)
                pad_width = [(diff // 2, (diff + 1) // 2)] + [
                    (0, 0) for _ in range(y.ndim - 1)
                ]
                y = numpy.pad(
                    y, pad_width=pad_width, constant_values=0, mode="constant"
                )
        return y


class BandpassPerturbation(object):
    """BandpassPerturbation class.

    Randomly dropout along the frequency axis.

    The original idea comes from the following:
        "randomly-selected frequency band was cut off under the constraint of
         leaving at least 1,000 Hz band within the range of less than 4,000Hz."
        (The Hitachi/JHU CHiME-5 system: Advances in speech recognition for
         everyday home environments using multiple microphone arrays;
         http://spandh.dcs.shef.ac.uk/chime_workshop/papers/CHiME_2018_paper_kanda.pdf)

    """

    def __init__(self, lower=0.0, upper=0.75, seed=None, axes=(-1,)):
        """Initialize class."""
        self.lower = lower
        self.upper = upper
        self.state = numpy.random.RandomState(seed)
        # x_stft: (Time, Channel, Freq)
        self.axes = axes

    def __repr__(self):
        """Return string with details of class."""
        return "{}(lower={}, upper={})".format(
            self.__class__.__name__, self.lower, self.upper
        )

    def __call__(self, x_stft, uttid=None, train=True):
        """Process call function."""
        if not train:
            return x_stft

        if x_stft.ndim == 1:
            raise RuntimeError(
                "Input in time-freq domain: " "(Time, Channel, Freq) or (Time, Freq)"
            )

        ratio = self.state.uniform(self.lower, self.upper)
        axes = [i if i >= 0 else x_stft.ndim - i for i in self.axes]
        shape = [s if i in axes else 1 for i, s in enumerate(x_stft.shape)]

        mask = self.state.randn(*shape) > ratio
        x_stft *= mask
        return x_stft


class VolumePerturbation(object):
    """Volume Perturbation class."""

    def __init__(self, lower=-1.6, upper=1.6, utt2ratio=None, dbunit=True, seed=None):
        """Initialize class."""
        self.dbunit = dbunit
        self.utt2ratio_file = utt2ratio
        self.lower = lower
        self.upper = upper
        self.state = numpy.random.RandomState(seed)

        if utt2ratio is not None:
            # Use the scheduled ratio for each utterances
            self.utt2ratio = {}
            self.lower = None
            self.upper = None
            self.accept_uttid = True

            with open(utt2ratio, "r") as f:
                for line in f:
                    utt, ratio = line.rstrip().split(None, 1)
                    ratio = float(ratio)
                    self.utt2ratio[utt] = ratio
        else:
            # The ratio is given on runtime randomly
            self.utt2ratio = None

    def __repr__(self):
        """Return string with details of class."""
        if self.utt2ratio is None:
            return "{}(lower={}, upper={}, dbunit={})".format(
                self.__class__.__name__, self.lower, self.upper, self.dbunit
            )
        else:
            return '{}("{}", dbunit={})'.format(
                self.__class__.__name__, self.utt2ratio_file, self.dbunit
            )

    def __call__(self, x, uttid=None, train=True):
        """Process call function."""
        if not train:
            return x

        x = x.astype(numpy.float32)

        if self.accept_uttid:
            ratio = self.utt2ratio[uttid]
        else:
            ratio = self.state.uniform(self.lower, self.upper)
        if self.dbunit:
            ratio = 10 ** (ratio / 20)
        return x * ratio


class NoiseInjection(object):
    """Add isotropic noise."""

    def __init__(
        self,
        utt2noise=None,
        lower=-20,
        upper=-5,
        utt2ratio=None,
        filetype="list",
        dbunit=True,
        seed=None,
    ):
        """Initialize class."""
        self.utt2noise_file = utt2noise
        self.utt2ratio_file = utt2ratio
        self.filetype = filetype
        self.dbunit = dbunit
        self.lower = lower
        self.upper = upper
        self.state = numpy.random.RandomState(seed)

        if utt2ratio is not None:
            # Use the scheduled ratio for each utterances
            self.utt2ratio = {}
            with open(utt2noise, "r") as f:
                for line in f:
                    utt, snr = line.rstrip().split(None, 1)
                    snr = float(snr)
                    self.utt2ratio[utt] = snr
        else:
            # The ratio is given on runtime randomly
            self.utt2ratio = None

        if utt2noise is not None:
            self.utt2noise = {}
            if filetype == "list":
                with open(utt2noise, "r") as f:
                    for line in f:
                        utt, filename = line.rstrip().split(None, 1)
                        signal, rate = soundfile.read(filename, dtype="int16")
                        # Load all files in memory
                        self.utt2noise[utt] = (signal, rate)

            elif filetype == "sound.hdf5":
                self.utt2noise = SoundHDF5File(utt2noise, "r")
            else:
                raise ValueError(filetype)
        else:
            self.utt2noise = None

        if utt2noise is not None and utt2ratio is not None:
            if set(self.utt2ratio) != set(self.utt2noise):
                raise RuntimeError(
                    "The uttids mismatch between {} and {}".format(utt2ratio, utt2noise)
                )

    def __repr__(self):
        """Return string with details of class."""
        if self.utt2ratio is None:
            return "{}(lower={}, upper={}, dbunit={})".format(
                self.__class__.__name__, self.lower, self.upper, self.dbunit
            )
        else:
            return '{}("{}", dbunit={})'.format(
                self.__class__.__name__, self.utt2ratio_file, self.dbunit
            )

    def __call__(self, x, uttid=None, train=True):
        """Process call function."""
        if not train:
            return x
        x = x.astype(numpy.float32)

        # 1. Get ratio of noise to signal in sound pressure level
        if uttid is not None and self.utt2ratio is not None:
            ratio = self.utt2ratio[uttid]
        else:
            ratio = self.state.uniform(self.lower, self.upper)

        if self.dbunit:
            ratio = 10 ** (ratio / 20)
        scale = ratio * numpy.sqrt((x**2).mean())

        # 2. Get noise
        if self.utt2noise is not None:
            # Get noise from the external source
            if uttid is not None:
                noise, rate = self.utt2noise[uttid]
            else:
                # Randomly select the noise source
                noise = self.state.choice(list(self.utt2noise.values()))
            # Normalize the level
            noise /= numpy.sqrt((noise**2).mean())

            # Adjust the noise length
            diff = abs(len(x) - len(noise))
            offset = self.state.randint(0, diff)
            if len(noise) > len(x):
                # Truncate noise
                noise = noise[offset : -(diff - offset)]
            else:
                noise = numpy.pad(noise, pad_width=[offset, diff - offset], mode="wrap")

        else:
            # Generate white noise
            noise = self.state.normal(0, 1, x.shape)

        # 3. Add noise to signal
        return x + noise * scale


class RIRConvolve(object):
    """RIR Convolve class."""

    def __init__(self, utt2rir, filetype="list"):
        """Initialize class."""
        self.utt2rir_file = utt2rir
        self.filetype = filetype

        self.utt2rir = {}
        if filetype == "list":
            with open(utt2rir, "r") as f:
                for line in f:
                    utt, filename = line.rstrip().split(None, 1)
                    signal, rate = soundfile.read(filename, dtype="int16")
                    self.utt2rir[utt] = (signal, rate)

        elif filetype == "sound.hdf5":
            self.utt2rir = SoundHDF5File(utt2rir, "r")
        else:
            raise NotImplementedError(filetype)

    def __repr__(self):
        """Return string with details of class."""
        return '{}("{}")'.format(self.__class__.__name__, self.utt2rir_file)

    def __call__(self, x, uttid=None, train=True):
        """Process call function."""
        if not train:
            return x

        x = x.astype(numpy.float32)

        if x.ndim != 1:
            # Must be single channel
            raise RuntimeError(
                "Input x must be one dimensional array, but got {}".format(x.shape)
            )

        rir, rate = self.utt2rir[uttid]
        if rir.ndim == 2:
            # FIXME(kamo): Use chainer.convolution_1d?
            # return [Time, Channel]
            return numpy.stack(
                [scipy.convolve(x, r, mode="same") for r in rir], axis=-1
            )
        else:
            return scipy.convolve(x, rir, mode="same")

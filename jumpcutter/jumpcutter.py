import itertools
import pathlib
from dataclasses import dataclass
import subprocess
from tempfile import TemporaryDirectory

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import math
from shutil import copyfile
import os


@dataclass
class JumpcutterParams:
    frame_rate: float
    sample_rate: int
    threshold: float
    silent_speed: float
    sounded_speed: float
    frame_margin: int = 1
    frame_quality: int = 3

    @property
    def speed_array(self):
        """Return a list of silent, sounded speed.

        This can be used like speed_array[is_sounded]"""
        return [self.silent_speed, self.sounded_speed]

    @property
    def samples_per_frame(self):
        return self.sample_rate / self.frame_rate


@dataclass
class AudioInfo:
    audio_data: list
    max_volume: float
    num_samples: int
    num_frames: int


class Jumpcutter:
    # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
    AUDIO_FADE_ENVELOPE_SIZE = 400

    def __init__(self, input_file, params: JumpcutterParams):
        self._input_file = input_file
        self._params = params
        self._temp_dir = TemporaryDirectory()
        self._temp = pathlib.Path(self._temp_dir.name)

    def cleanup(self):
        self._temp_dir.cleanup()

    # TODO split this method more
    # TODO determine frame count in advance -> maybe longer %06d
    def split_input_video(self):
        xprint("====> splitting input video")
        command = ["ffmpeg", "-hide_banner", "-i", self._input_file, "-qscale:v", str(self._params.frame_quality),
                   self._temp / "frame%06d.jpg"]
        subprocess.call(command, shell=False)

        command = ["ffmpeg", "-hide_banner", "-i", self._input_file, "-ab", "160k", "-ac", "2", "-ar",
                   str(self._params.sample_rate),
                   "-vn",
                   self._temp / "audio.wav"]
        subprocess.call(command, shell=False)

    def _load_audio_info(self):
        _, audio_data = wavfile.read(self._temp / "audio.wav")
        num_audio_samples = audio_data.shape[0]
        num_audio_frames = int(math.ceil(num_audio_samples / self._params.samples_per_frame))
        return AudioInfo(
            audio_data=audio_data,
            num_samples=num_audio_samples,
            max_volume=self._calculate_max_volume(audio_data),
            num_frames=num_audio_frames
        )

    def _find_sounded_frames(self, audio_info):
        is_frame_sounded = [False] * audio_info.num_frames
        for i in range(audio_info.num_frames):
            start = int(i * self._params.samples_per_frame)
            end = min(int((i + 1) * self._params.samples_per_frame), audio_info.num_samples)
            if self._calculate_max_volume(
                    audio_info.audio_data[start:end]) / audio_info.max_volume >= self._params.threshold:
                is_frame_sounded[i] = True
        return is_frame_sounded

    @staticmethod
    def _apply_frame_margin(include_frame, margin):
        return [
            any(
                include_frame[
                max(0, i - margin)
                # this is +1 because slicing i:i yields []
                :min(len(include_frame), i + 1 + margin)
                ]
            )
            for i in range(len(include_frame))
        ]

    @staticmethod
    def _audio_chunks_from_frames(include_frame):
        chunks = []
        start = 0
        for val, group in itertools.groupby(include_frame):
            length = len(list(group))
            chunks.append([start, start + length, val])
            start += length
        return chunks

    def analyze_audio(self, audio_info):
        return self._audio_chunks_from_frames(
            self._apply_frame_margin(
                self._find_sounded_frames(audio_info),
                self._params.frame_margin
            )
        )

    def _warp_audio(self, chunks, audio_info):
        outputAudioData = np.zeros((0, audio_info.audio_data.shape[1]))
        audio_chunks = []
        start = 0
        for chunk in chunks:
            audio_chunk = audio_info.audio_data[
                          int(chunk[0] * self._params.samples_per_frame)
                          :int(chunk[1] * self._params.samples_per_frame)
                          ]

            sFile = str(self._temp / "tempStart.wav")
            eFile = str(self._temp / "tempEnd.wav")
            wavfile.write(sFile, self._params.sample_rate, audio_chunk)
            with WavReader(sFile) as reader:
                with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                    tsm = phasevocoder(
                        reader.channels,
                        speed=self._params.speed_array[int(chunk[2])]
                    )
                    tsm.run(reader, writer)
            _, alteredAudioData = wavfile.read(eFile)
            outputAudioData = np.concatenate((outputAudioData, alteredAudioData / audio_info.max_volume))
            new_start = start + alteredAudioData.shape[0]
            audio_chunks.append([start, new_start])
            start = new_start
        return outputAudioData, audio_chunks

    def _apply_envelopes(self, audio_chunks, audio_data):
        xprint("====> rearranging frames")

        for start, end in audio_chunks:
            if end - start < self.AUDIO_FADE_ENVELOPE_SIZE:
                audio_data[start:end] = 0  # audio is less than 0.01 sec, let's just remove it.
            else:
                premask = np.arange(self.AUDIO_FADE_ENVELOPE_SIZE) / self.AUDIO_FADE_ENVELOPE_SIZE
                mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
                audio_data[start:start + self.AUDIO_FADE_ENVELOPE_SIZE] *= mask
                audio_data[end - self.AUDIO_FADE_ENVELOPE_SIZE:end] *= 1 - mask
        return audio_data

    def _rearrange_frames(self, chunks, audio_chunks):

        for (start_in, _, do_include), (start_out, end_out) in zip(chunks, audio_chunks):
            lastExistingFrame = None

            start_frame = int(math.ceil(start_out / self._params.samples_per_frame))
            end_frame = int(math.ceil(end_out / self._params.samples_per_frame))
            for outputFrame in range(start_frame, end_frame):
                input_frame = int(start_in + self._params.speed_array[int(do_include)] * (outputFrame - start_frame))
                didItWork = self.copyFrame(input_frame, outputFrame)
                # TODO what's this?
                if didItWork:
                    lastExistingFrame = input_frame
                else:
                    self.copyFrame(lastExistingFrame, outputFrame)

    def render_output(self, outputAudioData):
        xprint("====> rendering output video")

        wavfile.write(self._temp / "audioNew.wav", self._params.sample_rate, outputAudioData)

        command = ["ffmpeg", "-y", "-hide_banner", "-framerate", str(self._params.frame_rate), "-i",
                   self._temp / "newFrame%06d.jpg", "-i",
                   self._temp / "audioNew.wav", "-strict", "-2", self._input_to_output(self._input_file)]
        subprocess.call(command, shell=False)

    @staticmethod
    def _input_to_output(input_file):
        return "out.mkv"

    @staticmethod
    def _calculate_max_volume(audio_data):
        return max(
            float(np.max(audio_data)),
            -float(np.min(audio_data))
        )

    def copyFrame(self, inputFrame, outputFrame):
        src = str(self._temp / "frame{:06d}".format(inputFrame + 1)) + ".jpg"
        dst = str(self._temp / "newFrame{:06d}".format(outputFrame + 1)) + ".jpg"
        if not os.path.isfile(src):
            return False
        copyfile(src, dst)
        if outputFrame % 100 == 99:
            xprint(str(outputFrame + 1) + " time-altered frames saved.")
        return True

    def do_everything(self):
        self.split_input_video()
        audio_info = self._load_audio_info()
        chunks = self.analyze_audio(audio_info)
        audio_data, audio_chunks = self._warp_audio(chunks, audio_info)
        output_audio_data = self._apply_envelopes(audio_chunks, audio_data)
        self._rearrange_frames(chunks, audio_chunks)
        self.render_output(output_audio_data)


def xprint(msg):
    import sys
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)


params = JumpcutterParams(
    threshold=0.05,
    silent_speed=999999,
    sounded_speed=1,
    frame_rate=15,
    sample_rate=48000
)

cutter = Jumpcutter("input.mkv", params)
cutter.do_everything()
cutter.cleanup()

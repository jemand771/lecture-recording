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

    def notify_progress(self, *args, **kwargs):
        pass

    def run_ffmpeg(self, command):
        self.notify_progress(next=True)
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-progress", "pipe:1"
            ] + command,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                self.notify_progress(progress=1.0)
                break  # process is done
            if output:
                if output.startswith(b"out_time_ms"):
                    self.notify_progress(out_time_ms=output.decode("utf-8").split("=")[1])

    @property
    def _temp_input_file_path(self):
        return self._temp / f"input.{self._input_file.rsplit('.', 1)[-1]}"

    def remux_input_video(self):
        self.run_ffmpeg([
            "-i", self._input_file,
            "-filter:v", f"fps={self._params.frame_rate}",
            self._temp_input_file_path,
        ])

    # TODO split this method more
    # TODO determine frame count in advance -> maybe longer %06d
    def split_input_video(self):
        self.run_ffmpeg([
            "-i", self._temp_input_file_path,
            "-qscale:v", str(self._params.frame_quality),
            self._temp / "frame%06d.jpg"
        ])

        self.run_ffmpeg([
            "-i", self._temp_input_file_path,
            "-ab", "160k",
            "-ac", "2",
            "-ar", str(self._params.sample_rate),
            "-vn",
            self._temp / "audio.wav"
        ])

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
                # this is +1 because slicing i:i yields []
                include_frame[max(0, i - margin):min(len(include_frame), i + 1 + margin)]
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
        self.notify_progress(next=True)
        output_audio = np.zeros((0, audio_info.audio_data.shape[1]))
        audio_chunks = []
        start = 0
        for chunk in chunks:
            audio_start = int(chunk[0] * self._params.samples_per_frame)
            audio_end = int(chunk[1] * self._params.samples_per_frame)
            audio_chunk = audio_info.audio_data[audio_start:audio_end]

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
            output_audio = np.concatenate((output_audio, alteredAudioData / audio_info.max_volume))
            new_start = start + alteredAudioData.shape[0]
            audio_chunks.append([start, new_start])
            start = new_start
            self.notify_progress(progress=chunk[1]/chunks[-1][1])
        return output_audio, audio_chunks

    def _apply_envelopes(self, audio_chunks, audio_data):
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
        self.notify_progress(next=True)

        for (start_in, end_in, do_include), (start_out, end_out) in zip(chunks, audio_chunks):
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
            self.notify_progress(progress=end_in/chunks[-1][1])

    def render_output_wav(self, audio_data):
        wavfile.write(self._temp / "audioNew.wav", self._params.sample_rate, audio_data)

    def render_output(self):
        self.run_ffmpeg([
            "-framerate", str(self._params.frame_rate),
            "-i", self._temp / "newFrame%06d.jpg",
            "-i", self._temp / "audioNew.wav",
            "-strict",
            "-2",
            self._input_to_output(self._input_file)
        ])

    def _input_to_output(self, input_file):
        name, ext = input_file.rsplit(".", 1)
        return (
            f"{name}_cut"
            f"-{self._params.threshold:.2f}"
            f"-{self._params.sounded_speed}"
            f"-{self._params.silent_speed}"
            f".{ext}"
        )

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
        return True


class JumpcutterDriver(Jumpcutter):
    progress_hooks = []
    job_progress = [None] * 6
    current_job = -1
    full_length = None
    done = False
    # remux
    # split input video
    # load audio
    # warp audio
    # rearrange
    # render

    def notify_progress(self, *args, **kwargs):
        if self.done:
            return
        if kwargs.get("next") is True:
            if self.current_job >= 0:
                self.notify_progress(progress=1.0)
            self.current_job += 1
            if self.current_job == len(self.job_progress):
                self.done = True
                return
            self.notify_progress(progress=0.0)
        if isinstance(kwargs.get("progress"), float):
            self.job_progress[self.current_job] = kwargs.get("progress")
            print(self.job_progress)
        if isinstance(kwargs.get("out_time_ms"), str):
            out_time_ms = kwargs.get("out_time_ms")
            self.notify_progress(progress=round(float(out_time_ms) / 1000)/self.full_length)

    @staticmethod
    def get_length(input_file):
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_file
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        return round(float(result.stdout) * 1000)

    def do_everything(self):
        self.full_length = self.get_length(self._input_file)
        self.remux_input_video()
        self.split_input_video()
        audio_info = self._load_audio_info()
        chunks = self.analyze_audio(audio_info)
        audio_data, audio_chunks = self._warp_audio(chunks, audio_info)
        output_audio_data = self._apply_envelopes(audio_chunks, audio_data)
        self._rearrange_frames(chunks, audio_chunks)
        self.full_length = self.get_length(self._input_file)
        self.render_output_wav(output_audio_data)
        self.full_length = self.get_length(self._temp / "audioNew.wav")
        self.render_output()


def main():
    params = JumpcutterParams(
        threshold=0.05,
        silent_speed=999999,
        sounded_speed=1,
        frame_rate=15,
        sample_rate=48000
    )

    cutter = JumpcutterDriver("input.mkv", params)
    # cutter.full_length = cutter.get_length()
    # cutter.split_input_video()
    cutter.do_everything()
    cutter.cleanup()


if __name__ == "__main__":
    main()

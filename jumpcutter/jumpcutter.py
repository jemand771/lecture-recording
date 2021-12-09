from dataclasses import dataclass
import itertools
import os
import pathlib
import subprocess
from tempfile import TemporaryDirectory
import time

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from discord_webhook import DiscordEmbed, DiscordWebhook
import math
import numpy as np
from scipy.io import wavfile
from shutil import copyfile
import webdav3.client


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
            self.notify_progress(progress=chunk[1] / chunks[-1][1])
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
            self.notify_progress(progress=end_in / chunks[-1][1])

    def render_output_wav(self, audio_data):
        wavfile.write(self._temp / "audioNew.wav", self._params.sample_rate, audio_data)

    def render_output(self):
        self.run_ffmpeg([
            "-framerate", str(self._params.frame_rate),
            "-i", self._temp / "newFrame%06d.jpg",
            "-i", self._temp / "audioNew.wav",
            "-strict",
            "-2",
            self._temp / self._input_to_output(self._input_file)
        ])

    def _input_to_output(self, input_file):
        name, ext = str(pathlib.Path(input_file).name).rsplit(".", 1)
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


class ProgressHook:
    JOB_TITLES = [
        "Remuxing",
        "Generating frames",
        "Extracting audio",
        "Warping audio",
        "Arranging frames",
        "Rendering output",
        "Uploading original",
        "Uploading result"
    ]
    last_progress = [
        {
            "title": title,
            "percentage": None
        }
        for title
        in JOB_TITLES
    ]
    len_orig = None
    len_new = None
    size_orig = None
    size_new = None

    def _progress(self, progress_info):
        pass

    def progress(self, progress_info):
        self.last_progress = progress_info
        return self._progress(progress_info)


class JumpcutterDriver(Jumpcutter):
    current_job = -1
    full_length = None
    done = False
    module_dir = None

    def __init__(self, *args, **kwargs):
        self.progress_hooks = []
        self.job_progress = [None] * 8
        self._data = {}
        super().__init__(*args, **kwargs)

    # TODO notify for "nothing" before anything happens
    def notify_progress(self, *args, **kwargs):
        if self.done:
            return
        if kwargs.get("next") is True:
            if self.current_job >= 0:
                self.notify_progress(progress=1.0)
            self.current_job += 1
            self.notify_progress(progress=0.0)
        if isinstance(kwargs.get("progress"), float):
            self.job_progress[self.current_job] = kwargs.get("progress")
            for hook in self.progress_hooks:
                hook.progress(self.job_progress_fmt)
        if isinstance(kwargs.get("out_time_ms"), str):
            out_time_ms = kwargs.get("out_time_ms")
            self.notify_progress(progress=round(float(out_time_ms) / 1000) / self.full_length)

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

    @staticmethod
    def get_size(input_file):
        return os.path.getsize(input_file)

    def do_work(self):
        # this whole function is really fucking ugly but how else am I supposed to
        # transform my fancy stateless class into this stateful mess?
        if self.done:
            return
        if self.current_job == -1:
            self.full_length = self.get_length(self._input_file)
            for hook in self.progress_hooks:
                hook.len_orig = self.full_length
                hook.size_orig = self.get_size(self._input_file)
            self.remux_input_video()
        elif self.current_job == 0:
            self.split_input_video()
        # splitting the input video counts as two jobs - video and audio
        elif self.current_job == 2:
            audio_info = self._load_audio_info()
            self._data["chunks"] = self.analyze_audio(audio_info)
            audio_data, self._data["audio_chunks"] = self._warp_audio(self._data["chunks"], audio_info)
            self._data["output_audio_data"] = self._apply_envelopes(self._data["audio_chunks"], audio_data)
        elif self.current_job == 3:
            self._rearrange_frames(self._data["chunks"], self._data["audio_chunks"])
        elif self.current_job == 4:
            self.render_output_wav(self._data["output_audio_data"])
            self.full_length = self.get_length(self._temp / "audioNew.wav")
            for hook in self.progress_hooks:
                hook.len_new = self.full_length
            self.render_output()
            for hook in self.progress_hooks:
                hook.size_new = self.get_size(self._temp / self._input_to_output(self._input_file))
        elif self.current_job == 5:
            self.do_upload(pathlib.Path(self._input_file))
        elif self.current_job == 6:
            self.do_upload(
                self._temp / self._input_to_output(self._input_file),
                cut=True
            )
            self.cleanup()
            self.done = True
        else:
            raise RuntimeError(f"I don't know what to do for job index {self.current_job + 1}")

    def do_upload(self, source, cut=False):
        self.notify_progress(next=True)
        target_path = (
            pathlib.Path(os.environ.get("NEXTCLOUD_FOLDER"))
        )
        if cut:
            target_path /= "_cut"
        if self.module_dir:
            target_path /= self.module_dir
        target_path /= source.name
        self.get_upload_client().mkdir(str(target_path.parent).replace("\\", "/"))
        self.get_upload_client().upload_file(
            local_path=source,
            remote_path=str(target_path).replace("\\", "/"),
            progress=lambda current, total: self.notify_progress(progress=current / total)
        )
        self.notify_progress(progress=1.0)

    @staticmethod
    def get_upload_client():
        return webdav3.client.Client({
            "webdav_hostname": os.environ.get("NEXTCLOUD_URL"),
            "webdav_login": os.environ.get("NEXTCLOUD_USERNAME"),
            "webdav_password": os.environ.get("NEXTCLOUD_PASSWORD")
        })

    @property
    def job_progress_fmt(self):
        return [
            {
                "percentage": percentage,
                "title": title
            }
            for percentage, title
            in zip(
                self.job_progress,
                ProgressHook.JOB_TITLES
            )
        ]


class DiscordHook(ProgressHook):

    def __init__(self, webhook_url, title, course):
        self.hook = DiscordWebhook(webhook_url)
        self.title = title
        self.course = course
        self.sent = None
        self.last_sent = 0.0

    def _progress(self, progress_info):
        if self.qualify_send(progress_info):
            embed = self.build_embed(progress_info)
            self.execute(embed)

    def qualify_send(self, progress_info):
        if time.time() - self.last_sent > 1:
            return True
        perc_arr = [x["percentage"] for x in progress_info]
        if all(x is None for x in perc_arr):
            return True
        if [x for x in perc_arr if x is not None][-1] in (0.0, 1.0):
            return True
        return False

    def build_embed(self, progress_info):
        embed = DiscordEmbed()
        embed.add_embed_field(name="File", value=self.title)
        embed.add_embed_field(name="Course", value=self.course),
        embed.add_embed_field(name="_ _", value="_ _")
        embed.add_embed_field(name="Original length", value=self.format_duration(self.len_orig))
        embed.add_embed_field(name="Jumpcut length", value=self.format_duration(self.len_new))
        embed.add_embed_field(name="Length ratio", value=self.ratio(self.len_new, self.len_orig))
        embed.add_embed_field(name="Original size", value=self.format_size(self.size_orig))
        embed.add_embed_field(name="Jumpcut size", value=self.format_size(self.size_new))
        embed.add_embed_field(name="Size ratio", value=self.ratio(self.size_new, self.size_orig))
        embed.add_embed_field(name="Progress", value=self.build_progress_content(progress_info), inline=False)
        embed.set_color(self.get_color([x["percentage"] for x in progress_info]))
        embed.set_timestamp()
        embed.set_footer(text="Live Status")
        return embed

    @staticmethod
    def ratio(part, total):
        if part is None or total is None:
            return "N/A"
        r = float(100 * part) / float(total)
        return f"{round(r)}%"

    @staticmethod
    def get_color(percentages):
        blue = 255
        red = 255 * 2 ** 16
        yellow = 256 * (255 + 255 * 256)
        green = 255 * 256
        if all(x == 1.0 for x in percentages):
            return green
        if all(x is None for x in percentages):
            return yellow
        final_progress = [x for x in percentages if x is not None][-1]
        if final_progress == 1.0:
            return yellow
        return blue

    @staticmethod
    def build_progress_content(progress_info):
        max_title_len = max(len(x["title"]) for x in progress_info)
        lines = []
        for step in progress_info:
            percent_done = round((step["percentage"] or 0) * 100)
            lines.append(f"`{percent_done:3} %` {step['title'].ljust(max_title_len)}")
        return "\n".join(lines)

    @staticmethod
    def format_duration(duration):
        if duration is None:
            return "N/A"
        return time.strftime("%H:%M:%S", time.gmtime(round(duration / 1000)))

    @staticmethod
    def format_size(size):
        if size is None:
            return "N/A"
        for unit in ["", "K", "M", "G"]:
            if abs(size) < 1024.0:
                return f"{size:.1f} {unit}iB"
            size /= 1024
        return "N/A"

    def execute(self, embed):
        self.hook.embeds.clear()
        self.hook.add_embed(embed)
        if self.sent is None:
            self.sent = self.hook.execute()
        else:
            self.hook.edit(self.sent)
        self.last_sent = time.time()


class StdoutHook(ProgressHook):

    def _progress(self, progress_info):
        print()
        max_title_len = max(len(x["title"]) for x in progress_info)
        for step in progress_info:
            percent_done = round((step["percentage"] or 0) * 100)

            print(
                f"{percent_done:3} %"
                f" {step['title'].ljust(max_title_len)}"
                f" |{percent_done * '#'}{(100 - percent_done) * ' '}|"
            )

import pathlib
from dataclasses import dataclass
import subprocess
import sys
from tempfile import TemporaryDirectory

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import math
from shutil import copyfile, rmtree
import os


@dataclass
class JumpcutterParams:
    frame_rate: float
    sample_rate: int
    threshold: float
    silent_speed: float
    sounded_speed: float
    frame_margin: float = 1.0
    frame_quality: int = 3


class Jumpcutter:
    # smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
    AUDIO_FADE_ENVELOPE_SIZE = 400

    def __init__(self, input_file, settings: JumpcutterParams):
        self._input_file = input_file
        self._settings = settings
        self._temp_dir = TemporaryDirectory()
        self._temp = pathlib.Path(self._temp_dir.name)

    def cleanup(self):
        self._temp_dir.cleanup()

    # TODO split this method more
    # TODO determine frame count in advance -> maybe longer %06d
    def split_input_video(self):
        xprint("====> splitting input video")
        command = ["ffmpeg", "-hide_banner", "-i", INPUT_FILE, "-qscale:v", str(FRAME_QUALITY),
                   self._temp / "frame%06d.jpg"]
        subprocess.call(command, shell=False)

        command = ["ffmpeg", "-hide_banner", "-i", INPUT_FILE, "-ab", "160k", "-ac", "2", "-ar", str(SAMPLE_RATE),
                   "-vn",
                   self._temp / "audio.wav"]
        subprocess.call(command, shell=False)

    # TODO understand this
    def initial_audio_stuff(self):
        sampleRate, audioData = wavfile.read(self._temp / "audio.wav")
        audioSampleCount = audioData.shape[0]
        maxAudioVolume = self._calculate_max_volume(audioData)

        samplesPerFrame = sampleRate / frameRate

        audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))

        hasLoudAudio = np.zeros((audioFrameCount))

        for i in range(audioFrameCount):
            start = int(i * samplesPerFrame)
            end = min(int((i + 1) * samplesPerFrame), audioSampleCount)
            audiochunks = audioData[start:end]
            maxchunksVolume = float(self._calculate_max_volume(audiochunks)) / maxAudioVolume
            if maxchunksVolume >= SILENT_THRESHOLD:
                hasLoudAudio[i] = 1

        chunks = [[0, 0, 0]]
        shouldIncludeFrame = np.zeros((audioFrameCount))
        for i in range(audioFrameCount):
            start = int(max(0, i - FRAME_SPREADAGE))
            end = int(min(audioFrameCount, i + 1 + FRAME_SPREADAGE))
            shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
            if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i - 1]):  # Did we flip?
                chunks.append([chunks[-1][1], i, shouldIncludeFrame[i - 1]])

        chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i - 1]])
        return chunks[1:], audioData, samplesPerFrame, maxAudioVolume

    def rearrange_frames(self, chunks, audioData, samplesPerFrame, maxAudioVolume):
        xprint("====> rearranging frames")
        outputAudioData = np.zeros((0, audioData.shape[1]))
        outputPointer = 0

        lastExistingFrame = None
        for chunk in chunks:
            audioChunk = audioData[int(chunk[0] * samplesPerFrame):int(chunk[1] * samplesPerFrame)]

            sFile = str(self._temp / "tempStart.wav")
            eFile = str(self._temp / "tempEnd.wav")
            wavfile.write(sFile, SAMPLE_RATE, audioChunk)
            with WavReader(sFile) as reader:
                with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                    tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                    tsm.run(reader, writer)
            _, alteredAudioData = wavfile.read(eFile)
            leng = alteredAudioData.shape[0]
            endPointer = outputPointer + leng
            outputAudioData = np.concatenate((outputAudioData, alteredAudioData / maxAudioVolume))

            # outputAudioData[outputPointer:endPointer] = alteredAudioData/maxAudioVolume

            # smooth out transitiion's audio by quickly fading in/out

            if leng < self.AUDIO_FADE_ENVELOPE_SIZE:
                outputAudioData[outputPointer:endPointer] = 0  # audio is less than 0.01 sec, let's just remove it.
            else:
                premask = np.arange(self.AUDIO_FADE_ENVELOPE_SIZE) / self.AUDIO_FADE_ENVELOPE_SIZE
                mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
                outputAudioData[outputPointer:outputPointer + self.AUDIO_FADE_ENVELOPE_SIZE] *= mask
                outputAudioData[endPointer - self.AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1 - mask

            startOutputFrame = int(math.ceil(outputPointer / samplesPerFrame))
            endOutputFrame = int(math.ceil(endPointer / samplesPerFrame))
            for outputFrame in range(startOutputFrame, endOutputFrame):
                inputFrame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - startOutputFrame))
                didItWork = self.copyFrame(inputFrame, outputFrame)
                if didItWork:
                    lastExistingFrame = inputFrame
                else:
                    self.copyFrame(lastExistingFrame, outputFrame)

            outputPointer = endPointer
        return outputAudioData

    def render_output(self, outputAudioData):
        xprint("====> rendering output video")

        wavfile.write(self._temp / "audioNew.wav", SAMPLE_RATE, outputAudioData)

        command = ["ffmpeg", "-y", "-hide_banner", "-framerate", str(frameRate), "-i",
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
        stuff = self.initial_audio_stuff()
        output_audio_data = self.rearrange_frames(*stuff)
        self.render_output(output_audio_data)



# TODO input these
frameRate = 15
SAMPLE_RATE = 48000
SILENT_THRESHOLD = 0.05
FRAME_SPREADAGE = 1
NEW_SPEED = [999999, 1]  # silent, sounded
INPUT_FILE = "input.mkv"
FRAME_QUALITY = 3


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

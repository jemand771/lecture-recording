import subprocess
import sys

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
from scipy.io import wavfile
import numpy as np
import math
from shutil import copyfile, rmtree
import os


def getMaxVolume(s):
    maxv = float(np.max(s))
    minv = float(np.min(s))
    return max(maxv, -minv)


def copyFrame(inputFrame, outputFrame):
    src = TEMP_FOLDER + "/frame{:06d}".format(inputFrame + 1) + ".jpg"
    dst = TEMP_FOLDER + "/newFrame{:06d}".format(outputFrame + 1) + ".jpg"
    if not os.path.isfile(src):
        return False
    copyfile(src, dst)
    if outputFrame % 100 == 99:
        xprint(str(outputFrame + 1) + " time-altered frames saved.")
    return True


def inputToOutputFilename(filename):
    dotIndex = filename.rfind(".")
    return filename[:dotIndex] + "_ALTERED" + filename[dotIndex:]


def createPath(s):
    # assert (not os.path.exists(s)), "The filepath "+s+" already exists. Don't want to overwrite it. Aborting."
    deletePath(s)
    try:
        os.mkdir(s)
    except OSError:
        assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"


def deletePath(s):  # Dangerous! Watch out!
    try:
        rmtree(s, ignore_errors=False)
    except OSError:
        pass
        # print ("Deletion of the directory %s failed" % s)
        # print(ex)


# TODO input these
frameRate = 15
SAMPLE_RATE = 48000
SILENT_THRESHOLD = 0.05
FRAME_SPREADAGE = 1
NEW_SPEED = [999999, 1]  # silent, sounded
INPUT_FILE = "input.mkv"
FRAME_QUALITY = 3

# TODO create output file name
OUTPUT_FILE = inputToOutputFilename(INPUT_FILE)

TEMP_FOLDER = "TEMP"
# smooth out transitiion's audio by quickly fading in/out (arbitrary magic number whatever)
AUDIO_FADE_ENVELOPE_SIZE = 400

createPath(TEMP_FOLDER)


def xprint(msg):
    import sys
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)


xprint("====> splitting input video")
command = ["ffmpeg", "-hide_banner", "-i", INPUT_FILE, "-qscale:v", str(FRAME_QUALITY), TEMP_FOLDER + "/frame%06d.jpg"]
subprocess.call(command, shell=False)

command = ["ffmpeg", "-hide_banner", "-i", INPUT_FILE, "-ab", "160k", "-ac", "2", "-ar", str(SAMPLE_RATE), "-vn",
           TEMP_FOLDER + "/audio.wav"]

subprocess.call(command, shell=False)

sampleRate, audioData = wavfile.read(TEMP_FOLDER + "/audio.wav")
audioSampleCount = audioData.shape[0]
maxAudioVolume = getMaxVolume(audioData)

samplesPerFrame = sampleRate / frameRate

audioFrameCount = int(math.ceil(audioSampleCount / samplesPerFrame))

hasLoudAudio = np.zeros((audioFrameCount))

for i in range(audioFrameCount):
    start = int(i * samplesPerFrame)
    end = min(int((i + 1) * samplesPerFrame), audioSampleCount)
    audiochunks = audioData[start:end]
    maxchunksVolume = float(getMaxVolume(audiochunks)) / maxAudioVolume
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
chunks = chunks[1:]

outputAudioData = np.zeros((0, audioData.shape[1]))
outputPointer = 0

xprint("====> rearranging frames")

lastExistingFrame = None
for chunk in chunks:
    audioChunk = audioData[int(chunk[0] * samplesPerFrame):int(chunk[1] * samplesPerFrame)]

    sFile = TEMP_FOLDER + "/tempStart.wav"
    eFile = TEMP_FOLDER + "/tempEnd.wav"
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

    if leng < AUDIO_FADE_ENVELOPE_SIZE:
        outputAudioData[outputPointer:endPointer] = 0  # audio is less than 0.01 sec, let's just remove it.
    else:
        premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE) / AUDIO_FADE_ENVELOPE_SIZE
        mask = np.repeat(premask[:, np.newaxis], 2, axis=1)  # make the fade-envelope mask stereo
        outputAudioData[outputPointer:outputPointer + AUDIO_FADE_ENVELOPE_SIZE] *= mask
        outputAudioData[endPointer - AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1 - mask

    startOutputFrame = int(math.ceil(outputPointer / samplesPerFrame))
    endOutputFrame = int(math.ceil(endPointer / samplesPerFrame))
    for outputFrame in range(startOutputFrame, endOutputFrame):
        inputFrame = int(chunk[0] + NEW_SPEED[int(chunk[2])] * (outputFrame - startOutputFrame))
        didItWork = copyFrame(inputFrame, outputFrame)
        if didItWork:
            lastExistingFrame = inputFrame
        else:
            copyFrame(lastExistingFrame, outputFrame)

    outputPointer = endPointer

xprint("====> rendering output video")

wavfile.write(TEMP_FOLDER + "/audioNew.wav", SAMPLE_RATE, outputAudioData)

command = ["ffmpeg", "-y", "-hide_banner", "-framerate", str(frameRate), "-i", TEMP_FOLDER + "/newFrame%06d.jpg", "-i",
           TEMP_FOLDER + "/audioNew.wav", "-strict", "-2", OUTPUT_FILE]
subprocess.call(command, shell=False)

deletePath(TEMP_FOLDER)

# Required third party packages: whisper and openai
# See instructions for setup here: https://github.com/openai/whisper#setup
#   - You can use the below command to pull the repo and install dependencies, then just put this script in the repo directory:
#     pip install git+https://github.com/openai/whisper.git
#
# If you have issues with 'file not found' you may need to fix the 'ffmpeg' install.
# There are tutorials online, but basically you need to make sure it is added to your system path

import whisper
import io
import time
import os
import json
import pathlib
import openai

openai.api_key = 'YOUR-OPENAI-API-KEY-HERE'
# Choose model to use by uncommenting
# modelName = "tiny.en"
modelName = "base.en"
# modelName = "small.en"
# modelName = "medium.en"
# modelName = "large-v2"

# Other Variables
# (bool) Whether to export the segment data to a json file. Will include word level timestamps if word_timestamps is True.
exportTimestampData = False

#  ----- Select variables for transcribe method  -----
# audio: path to audio file
verbose = False  # (bool): Whether to display the text being decoded to the console. If True, displays all the details, If False, displays minimal details. If None, does not display anything
language = "english"  # Language of audio file
# (bool): Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.
word_timestamps = False
# initial_prompt="" # (optional str): Optional text to provide as a prompt for the first window. This can be used to provide, or "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns to make it more likely to predict those word correctly.

#  -------------------------------------------------------------------------
print(f"Using Model: {modelName}")
filePath = input("Path to File Being Transcribed: ")
filePath = filePath.strip("\"")
if not os.path.exists(filePath):
    print("Problem Getting File...")
    input("Press Enter to Exit...")
    exit()

fileNameStem = pathlib.Path(filePath).stem
outputFolder = f"{fileNameStem} Output"

# If output folder does not exist, create it
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
    print(f"\nCreated Output Folder named {outputFolder}.\n")

# Get filename stem using pathlib (filename without extension)

resultFileName = f"{fileNameStem}.txt"
jsonFileName = f"{fileNameStem}.json"

model = whisper.load_model(modelName)
start = time.time()

#  ---------------------------------------------------
result = model.transcribe(audio=filePath, language=language,
                          word_timestamps=word_timestamps, verbose=verbose)
#  ---------------------------------------------------

# Save transcription text to file
print("\nWriting transcription to file...")
with open(os.path.join(outputFolder, resultFileName), "w", encoding="utf-8") as file:
    file.write(result["text"])
print(
    f"Finished writing transcription file to {outputFolder}\\{resultFileName}")
print("Starting transcription summary")


def split_text_into_chunks(text, max_length=4000):
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        pos = text.rfind('.', 0, max_length)
        if pos == -1:
            pos = max_length
        else:
            pos += 1  # Include the full stop in the chunk.
        chunk = text[:pos]
        chunks.append(chunk)
        text = text[pos:].lstrip()  # Remove leading spaces from next chunk

    return chunks


with open(os.path.join(outputFolder, resultFileName), "r") as farts:
    page_text = farts.read()

chunks = split_text_into_chunks(page_text)
summary_text = ''

# Starting summarisation of transcription #

count = 1
for chunk in chunks:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful lecture summarizer."},
            {"role": "user", "content": f"Please Summarize this keeping important details: {chunk}"},
        ],)
    text_summary = response["choices"][0]["message"]["content"]
    print(f"Chunk {count} summarised.")
    count += 1
    summary_text += text_summary + "\n"
    text_summary_file = f"{fileNameStem}_summary.txt"
    with open(os.path.join(outputFolder, text_summary_file), "w", encoding="utf-8") as file:
        file.write(summary_text)

# Save the segments data to json file
# if word_timestamps == True:
if exportTimestampData == True:
    print("\nWriting segment data to file...")
    with open(os.path.join(outputFolder, jsonFileName), "w", encoding="utf-8") as file:
        segmentsData = result["segments"]
        json.dump(segmentsData, file, indent=4)
    print("Finished writing segment data file.")

end = time.time()
elapsed = float(end - start)

elapsedMinutes = str(round(elapsed/60, 2))
print(f"\nElapsed Time With {modelName} Model: {elapsedMinutes} Minutes")
print(
    f"Summary of transcription written to {outputFolder}\\{text_summary_file}")
input("Press Enter to exit...")
exit()

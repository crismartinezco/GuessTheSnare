# 🥁 GuessTheSnare: Sonic Similarity-Based Recommendation Model for Drums

## Overview
This project recommends sounds from a user-provided sound library that are sonically (psychoacoustically) similar to the sounds extracted from a target user-provided loop. This is done by extracting and classifying the onsets of the provided loop into kick, hh and snare and comparing the acoustic characteristics of said onsets to those of the sounds in the provided sound library. It utilizes onset detection, spectral centroid and flatness, and K-Means clustering. Check the JUPYTER version, since the py file is still being worked on. Here an example is provided for snares.

![Image](https://github.com/user-attachments/assets/f422ea22-48ef-4b54-a0dd-bbe695fcfe03)

## Features
- **Onset Detection**: Identifies transient events in the audio.
- **Feature Extraction**: Computes spectral flatness and spectral centroid.
- **Clustering**: Uses K-Means clustering to group similar percussive elements.
- **Audio Concatenation**: Creates new audio sequences by concatenating extracted percussive events.
- **Visualization**: Plots waveforms with detected onsets and clustering results.

## Dependencies
This project requires the following Python libraries:
```bash
pip install numpy scipy librosa matplotlib pandas scikit-learn soundfile
```
## Usage
1. **Load your loop and your sound library**
   - The script loads `BeatClean.wav` and detects onsets. Change to \your\audio\path.

2. **Extract onset features**
   - Spectral flatness and spectral centroid are computed for each onset.

3. **Clustering onsets**
   - K-Means clustering groups onsets into three categories (kick, snare, hi-hat).

4. **Visualize clustering results**
   - The script generates scatter plots of extracted features and overlays detected onsets on the waveform.

5. **Listen, compare and decide! ;)**
   - Extracted percussive sounds are played. Infor regarding name of the chosen sound sample from your library is provided.

Run the script:
```bash
python main.py
```

## Example Output
- **Waveform visualization with detected onsets**
- **Scatter plot of clustered percussive elements**
- **Concatenated snare, kick, and hi-hat audio files**

## Author
**Cristhiam Martínez**  
Email: [crismartinez@t-online.de](mailto:crismartinez@t-online.de)

## License
This project is licensed under the MIT License.


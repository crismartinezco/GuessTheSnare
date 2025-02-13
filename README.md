# 🥁 GuessTheSnare: Analysis and Clustering of Drum Sounds

## Overview
This project focuses on the analysis of drum sounds in an audio file. It utilizes onset detection, feature extraction, and clustering to separate different percussive elements (e.g., kick, snare, hi-hat). The clustered onsets are further processed to generate concatenated audio samples for each percussion type.

## Features
- **Onset Detection**: Identifies transient events in the audio.
- **Feature Extraction**: Computes spectral flatness and spectral centroid.
- **Clustering**: Uses K-Means clustering to group similar percussive elements.
- **Audio Concatenation**: Creates new audio sequences by concatenating extracted percussive events.
- **Visualization**: Plots waveforms with detected onsets and clustering results.

## Dependencies
This project requires the following Python libraries:
```bash
pip install numpy scipy librosa matplotlib pandas scikit-learn adjustText soundfile
```

## File Structure
```
GuessTheSnare/
├── main.py                # Main script for onset detection, clustering, and concatenation
├── BeatClean.wav          # Example audio file
├── concatenated_snare.wav # Output file containing extracted snare onsets
├── Cluster1.wav           # Click-marked audio from cluster 1
├── Cluster2.wav           # Click-marked audio from cluster 2
├── Cluster3.wav           # Click-marked audio from cluster 3
├── README.md              # Project documentation
```

## Usage
1. **Load the audio file**
   - The script loads `BeatClean.wav` and detects onsets. Change to \your\audio\path.

2. **Extract onset features**
   - Spectral flatness and spectral centroid are computed for each onset.

3. **Clustering onsets**
   - K-Means clustering groups onsets into three categories (kick, snare, hi-hat).

4. **Visualize clustering results**
   - The script generates scatter plots of extracted features and overlays detected onsets on the waveform.

5. **Concatenate and export clustered onsets**
   - Extracted percussive sounds are concatenated and saved as new audio files.

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


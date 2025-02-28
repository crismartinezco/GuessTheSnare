# 🥁 GuessTheSnare: Sonic Similarity-Based Recommendation Model for Drums

## Overview
This project recommends sounds from a user-provided sound library that are sonically (psychoacoustically) similar to the sounds extracted from a target user-provided loop. This is done by extracting and classifying the onsets of the provided loop into kick, hh and snare and comparing the acoustic characteristics of said onsets to those of the sounds in the provided sound library. It utilizes onset detection, spectral centroid and flatness, and K-Means clustering. Check the JUPYTER version, since the py file is still being worked on. Here an example is provided for snares.

![Extracted Onsets]([extracted onsets.png](https://github.com/crismartinezco/GuessTheSnare/blob/main/Clustering.png))


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


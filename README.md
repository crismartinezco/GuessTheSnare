# GuessTheSnare
Python-based machine learning tool for music producers to explore percussion sound libraries according to sound characteristics mapped to psychoacoustical parameters. In this first stage, the data base consists of 45 snare sounds from which the spectral bandwidth, spectral centroid, spectral flatness and spectral rolloff. These snare sounds were downloadad from the website www.noiiz.com and are used for academic purposes. I take no credit in the recording of the samples. Pipeline of the algorithm:

    1. Load the drum loop to be analyzed
    2. Through beat detection and a K-Means algorithm, separate hi hat, kick dru and snares
    3. Extract the snare samples
    4. Analyze spectral spectral features from the extracted samples
    5. Compare this acoustic features with the ones extracted from the sound data set, and based on the euclidean distance, look for the samples that presents the closest values to the ones from the loops

For further development more acoustic features shall be implemented, as well as a data augmentation module to add compressors and reverbs to the snare sounds from the data set.

Beatclean.wav and snares.zip are sample files with which the tool has been tested (they are implemented in the code).

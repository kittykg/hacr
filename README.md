# HACR -- Hybrid Architecture for Concept Reasoning

4th year individual project

Dataset: TVQA+

Tool stack:

- Object detection: `torch`, `torchvision`

- Face recognition: `dlib`, `face-recognition-models`, `opencv-python`,
  `scikit-learn`

- Language processing: `nltk`, `spacy`, `pyinflect`

- Logic based learning: `ILASP`

- ASP: `clingo`

## Pipeline

Base:

- Question parser

- ILASP Learner

- ASP solver

----

Add-on components:

- Object detection

- Face identifier


## Collected data

- `face_collection.npz`: Collected faces from all video frames. Includes 2
  arrays. This file is a lot smaller, so good for re-train face k-means cluster.
    * `faces`: Face encoding of all faces.
    * `labels`: Label for each face.

- `face_collection_v2.npz`: Collected faces from all video frames. Includes 3
  arrays. This file is 3GB and will take a long time to load. But it includes
  all the faces so it's useful to see the faces and their corresponding labels.
    * `faces`: All faces in 150 x 150. Could be plotted.
    * `encoded_faces`: Face encoding of all faces. This is equivalent to `faces`
      in `face_collection.npz`, although there's numerical difference.
    * `labels`: Label for each face.

- `train_hold.json`: Questions that could be learned and answered with the full
  pipeline

- `hold_questions.json`: All hold questions that could be learned and answered
  with ground truth bounding boxes.

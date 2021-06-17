# Collected NPZ

This folder contains the collected data for training our face classifier and
abrupt transition detection model. Note that `face_collection_v2.npz` is too big
to be uploaded onto GitHub, is only store locally.

- `face_collection.npz`: Collected faces from all video frames. Includes 2
  arrays. This file is a lot smaller, so good for re-train face k-means cluster.
    * `faces`: Face encoding of all faces.
    * `labels`: Label for each face.

- `face_collection_v2.npz`: Collected faces from all video frames. Includes 3
  arrays. This file is 3GB and will take a long time to load. But it includes
  all the faces so it's useful to see the faces and their corresponding labels.
  This file is saved on GoogleDrive for size reason. Please download it from
  [here](https://drive.google.com/file/d/1ft6-4_uZ5SNS_KJSKahxO2zhy-v34yx6/view?usp=sharing).
    * `faces`: All faces in 150 x 150. Could be plotted.
    * `encoded_faces`: Face encoding of all faces. This is equivalent to `faces`
      in `face_collection.npz`, although there's numerical difference.
    * `labels`: Label for each face.

- `transition_collection_all.npz`: Collected dissimilarity score with SAD,
  Hist-All and Hist-Reg
    * `pix_abs`: All Sum of Absolute Difference (SAD) scores for 4364926 pairs
      of consecutive frames in total.
    * `hist_all`: All Hist-All scores for 1853275 pairs of consecutive frames in
      total.
    * `hist_reg`: All Hist-Reg scores for 1853275 pairs of consecutive frames in
      total.

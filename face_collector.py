import json

import numpy as np
from tqdm import tqdm

from image_processing import ObjectDetector, match_person_bounding_boxes
from json_parser import get_img_ids

if __name__ == '__main__':
    od_threshold = 0.7
    train_subset_size = 10

    root_folder = '../TVQA_frames/frames_hq/bbt_frames/'
    train_json_file_path = '../tvqa_plus_train_prettified.json'
    npz_file_path = './face_collection.npz'

    od = ObjectDetector()

    encoded_faces = None
    labels = None

    with open(train_json_file_path) as f:
        all_json_data = json.load(f)

    for i in range(train_subset_size):
        data = all_json_data[i]
        vid_folder = root_folder + data['vid_name'] + '/'

        # Get specific images for this video
        specific_imgs = get_img_ids(data)

        # Get all QA objects
        qa_objects = od.get_rcnn_qa_objects(vid_folder, od_threshold,
                                            specific_imgs)

        # Match person grounding
        matched_pair = match_person_bounding_boxes(qa_objects, data)

        # Get human faces and ground truth labels
        matched_pair_qa = [o for o, _, _ in matched_pair]
        faces, idx = od.get_human_faces(matched_pair_qa)

        if len(faces) == 0:
            continue

        enc = od.encode_faces(faces)
        if encoded_faces is None:
            encoded_faces = enc
        else:
            encoded_faces = np.concatenate((encoded_faces, enc))

        qa_labels = np.array([l for _, l, _ in matched_pair])[idx]
        if labels is None:
            labels = qa_labels
        else:
            labels = np.concatenate((labels, qa_labels))

    np.savez_compressed(npz_file_path, faces=encoded_faces, labels=labels)

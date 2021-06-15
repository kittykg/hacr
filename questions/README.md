# Questions

All the training datasets are in this directory.

## Holding questions

**Fields** (all compatible with TVQA+):

* `a0` - `a4` (`str`): choices of the question.
* `answer_idx` (`str`): ground truth answer index.
* `bbox` (`dict[str -> arr]`): bounding boxes. The keys are the timestamps in string and the value is an array of bounding boxes object at that time.
* `q` (`str`): question.
* `qid` (`int`): unique question identifier.
* `ts` (`arr[float]`): ground truth relevant time span that answers the questions. An array of two floats inidactitng the start and end of the time span.
* `vid_name` (`str`): the video name for the question.

**Files**:

* `hold_questions.json`: All the 'holding' questions from TVQA+ train set with the format 'What is *someone* holding...' and can be parsed by our current question parsing process. 885 questions in the file.

* `train_hold.json`: The 'holding' questions from TVQA+ train set that the answer object can be put into COCO category with syno/hypo/hyper-nym check. 91 questions in the file.

## Entering questions

**Fields** compatible with TVQA+:

* `a0` - `a4` (`str`): choices of the question.
* `answer_idx` (`str`): ground truth answer index.
* `q` (`str`): question.
* `qid` (`int`): unique question identifier.
* `ts` (`arr[float]`): ground truth relevant time span that answers the questions. An array of two floats inidactitng the start and end of the time span.
* `vid_name` (`str`): the video name for the question.

**Fields** created by us (not compatible with TVQA+):

* `in_camera` (`dict[str -> arr]`): a record of the characters and their appearance time span. The key is the name of the character and the value is a list of time spans.
* `initial_in_scene`: the characters that are already in the current scene when the relavant ground truth relevant time span starts.
* `scene_change_pairs` (`arr[arr[int]]`): list of abrupt transitions in the ground truth time span. The abrupt transition is represented as an array with two ints.

**Files**:

* `enter.json`: Our hand-crafted questions for learning 'entering the scene', having the form of 'Who enters the scene...'. 10 questions in the file.



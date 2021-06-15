# Test data

This folder contains the datasets for evalution of individual components and overal pipeline.

* `abrupt_transition.json`: 10 video clips with all abrupt transitions annotated from the TVQA+ train set. Used to evaluate our abrupt transition detection.

* `val_enter.json`: 3 questions that can be categorised as 'Who enters the scene...' from the TVQA+ val set. The wording of the questions in this file has been changed by us and is different from the original wording. The rest is unchanged. Used to evalute our rule leaning and overall pipeline on the task of 'entering' learning.

* `val_hold_all.json`: 104 questions of the form 'What is *someone* holding...' from the TVQA+ val set. Used to evaluate our rule learning and overall pipeline on the task of 'holding' learning.

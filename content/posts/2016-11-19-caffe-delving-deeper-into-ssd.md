---
title: Delving deep into the SSD detector
date: 2016-11-19T20:51:39-03:00
draft: false
tags: ["caffe", "c++", "deep-learning"]
params:
  math: true
---

> This post was originally an internal guide. The title was a reference to the paper ["Return of the Devil in the Details: Delving Deep into Convolutional Nets"][4], by Chatfield et al.

This post delves deeper into the SSD detector, the [Single-Shot Multibox
Detector][1], implemented in [Caffe][2]. Usually, SSD and other detectors apply
a non-maximal suppression that only outputs the score of the object with the
highest score for a given anchor. In my case, I also needed the scores of the
other potential object classes, that is, the full softmax vector. We want to
extract the scores for all the objects contained in the dataset, not only the
object with the highest score. This post details, based on this [pull
request][3], what is required to modify the Caffe implementation to get the
whole softmax vector.

![SSD Detector](/images/ssd_detector.png)

## Gory Details

For each box out of \(k\) at a given location, the detector outputs a class \(c\) and 4 coordinate offsets \((cx, cy, w, h)\) This results in a total of \((c+4)k\) filters applied around each location. \((cx, cy)\) are coordinates from the center of the bounding box. There are \(k\) windows at a given location because there are feature maps in \(k\) scales. The authors report "6 default boxes per feature map location" i.e, \( \{1, 2, 3, \frac{1}{2}, \frac{1}{3} \} + \{\sqrt{s_k s_{k+1}}\} \). Note that scales are for each feature map location, and not a scale corresponding to each feature map.

After the fully-connected `fc7`, there are two convolutions, `conv6_1` and `conv6_2` of 1x1x256 and 3x3x512 respectively. These are all feature maps with location + class predictions. From the SSD_300x300.prototxt definition:

    * conv4_3
    * fc7
    * conv6_2 / conv8_2 (diagram)
    * conv7_2 / conv9_2
    * conv8_2 / conv10_2
    * pool6 (which is conv8_2 averaged pooled) / conv11_2

Here, it seems there is a a mismatch with what is depicted in Figure 1, possibly because this is the 300x300 resolution. I included the corresponding missing layers from the diagram. Also, the authors report that for `conv4_3`, `conv10_2`, and `conv11_2` only 4 default boxes (does not use aspect ratios 1/3 and 3).

So far, we’ve seen that each loc layer predicts 4 coordinates offsets. Thus, to fill our heatmaps with the prediction scores, we also need the grid to retrieve the final bounding box position. We can retrieve the grid from the `net.blobs['*_mbox_priorbox']`. For instance:

    conv6_2: (1, 512, 10, 10)
    conv6_2_mbox_priorbox: (1, 2, 2400)

which is given by \(10 * 10 * 6 \text{scales} * 4 \text{coordinates} = 2400 \).

Another example:

    conv7_2 (1, 256, 5, 5)
    conv7_2_mbox_priorbox (1, 2, 600)

given by \( 5 * 5 * 6 * 4 = 600 \).

Notice that it has 2 dimensions. The first dimension comprises the coordinates, while the second dimension are variances, which is always set to 0.1 and 0.2, alternating.

In short,

* `*_mbox_priorbox`: grid
* `*_mbox_loc_flat`: coordinate offsets (matches the `mbox_priorbox`'s last dimension)
* `mbox_conf`: prediction scores. It doesn't match these dimensions because it is the number of classes plus some gibberish that I still don’t know (1206 in total, for each loc layer, plus the other dimensions regarding the size of the feature map).

## Retrieving all probability scores for each bounding box

I can't use only `mbox_priorbox`, `loc_flat`, and `mbox_conf`, because it still hasn't been non-maximally supressed. The layer that does that is `DetectionOut`. Thus, you need to do the following modifications to include the other predictions:

```cpp
---
 src/caffe/layers/detection_output_layer.cpp | 21 +++++++++++++--------
 src/caffe/layers/detection_output_layer.cu | 24 +++++++++++++++---------
 2 files changed, 28 insertions(+), 17 deletions(-)

diff --git a/src/caffe/layers/detection_output_layer.cpp b/src/caffe/layers/detection_output_layer.cpp
index 50f5e7a..7881575 100644
--- a/src/caffe/layers/detection_output_layer.cpp
+++ b/src/caffe/layers/detection_output_layer.cpp
@@ -247,7 +247,7 @@ void DetectionOutputLayer<Dtype>::Forward_cpu(

  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  - top_shape.push_back(7);
  + top_shape.push_back(7 + num_classes_);
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = 1;

  @@ -286,17 +286,22 @@ void DetectionOutputLayer<Dtype>::Forward_cpu(
    << "Cannot find label: " << label << " in the label map.";
    CHECK_LT(name_count_, names_.size());
  }
  + int mconst = 7 + num_classes_; // mconst
  for (int j = 0; j < indices.size(); ++j) {
    int idx = indices[j];
    - top_data[count * 7] = i;
    - top_data[count * 7 + 1] = label;
    - top_data[count * 7 + 2] = scores[idx];
    + top_data[count * mconst] = i;
    + top_data[count * mconst + 1] = label;
    + top_data[count * mconst + 2] = scores[idx];
    NormalizedBBox clip_bbox;
    ClipBBox(bboxes[idx], &clip_bbox);
    - top_data[count * 7 + 3] = clip_bbox.xmin();
    - top_data[count * 7 + 4] = clip_bbox.ymin();
    - top_data[count * 7 + 5] = clip_bbox.xmax();
    - top_data[count * 7 + 6] = clip_bbox.ymax();
    + top_data[count * mconst + 3] = clip_bbox.xmin();
    + top_data[count * mconst + 4] = clip_bbox.ymin();
    + top_data[count * mconst + 5] = clip_bbox.xmax();
    + top_data[count * mconst + 6] = clip_bbox.ymax();
    + for (int c = 0; c < num_classes_; ++c){
    + const vector<float>& tempscores = conf_scores.find(c)->second;
    + top_data[count * mconst + 7+c] = tempscores[idx]; //all_conf_scores[c][idx];conf_scores[c][idx];
  + }
  if (need_save_) {
    NormalizedBBox scale_bbox;
    ScaleBBox(clip_bbox, sizes_[name_count_].first,
diff --git a/src/caffe/layers/detection_output_layer.cu b/src/caffe/layers/detection_output_layer.cu
index a07181c..55d6514 100644
--- a/src/caffe/layers/detection_output_layer.cu
+++ b/src/caffe/layers/detection_output_layer.cu
@@ -116,9 +116,9 @@ void DetectionOutputLayer<Dtype>::Forward_gpu(
  }
}

- vector<int> top_shape(2, 1);
+ vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
- top_shape.push_back(7);
+ top_shape.push_back(7 + num_classes_);
if (num_kept == 0) {
  LOG(INFO) << "Couldn't find any detections";
  top_shape[2] = 1;
@@ -157,17 +157,23 @@ void DetectionOutputLayer<Dtype>::Forward_gpu(
  << "Cannot find label: " << label << " in the label map.";
  CHECK_LT(name_count_, names_.size());
}
+ const int nitems = 7 + num_classes_; // mconst
+ int mconst = 7 + num_classes_; // mconst
  for (int j = 0; j < indices.size(); ++j) {
    int idx = indices[j];
-   top_data[count * 7] = i;
-   top_data[count * 7 + 1] = label;
-   top_data[count * 7 + 2] = scores[idx];
+   top_data[count * mconst] = i;
+   top_data[count * mconst + 1] = label;
+   top_data[count * mconst + 2] = scores[idx];
    NormalizedBBox clip_bbox;
    ClipBBox(bboxes[idx], &clip_bbox);
-   top_data[count * 7 + 3] = clip_bbox.xmin();
-   top_data[count * 7 + 4] = clip_bbox.ymin();
-   top_data[count * 7 + 5] = clip_bbox.xmax();
-   top_data[count * 7 + 6] = clip_bbox.ymax();
+   top_data[count * mconst + 3] = clip_bbox.xmin();
+   top_data[count * mconst + 4] = clip_bbox.ymin();
+   top_data[count * mconst + 5] = clip_bbox.xmax();
+   top_data[count * mconst + 6] = clip_bbox.ymax();
+   for (int c = 0; c < num_classes_; ++c){
+     const vector<float>& scores = conf_scores.find(c)->second;
+     top_data[count * mconst + 7 + c] = tempscores[idx]; //all_conf_scores[c][idx];conf_scores[c][idx];
+   }
    if (need_save_) {
      NormalizedBBox scale_bbox;
      ScaleBBox(clip_bbox, sizes_[name_count_].first,
```

The order is the following:

```python
# Parse the outputs.
det_label = detections[0,0,:,1]
det_conf = detections[0,0,:,2]
det_xmin = detections[0,0,:,3]
det_ymin = detections[0,0,:,4]
det_xmax = detections[0,0,:,5]
det_ymax = detections[0,0,:,6]
all_scores = detections[0,0,:,7:] # shape (201,0)
```

Notice that, if background has the highest probability, it won't be passed to the `det_conf` and `det_label` variables. It passes the second best prediction, which would be of an object and it will has a lower score. The reasoning is such that it presumed that the majority of the windows are background, and only few places will have objects. So, we only propagate the predictions of the objects.

## Remarks

* The `loc` layers are connected in cascade instead of in parallel;
* Class imbalance given that only a small number of windows match the ground-truth box. Thus, the authors mine hard negative (top score ones) respecting a ratio of 3:1 positive windows;
* Certain feature maps have different scales, thus they need to use a L2-norm to scale the feature norm at each location (see "`conv4_3` has a different feature scale").

## Shape of location and softmax

```ipython
In [19]: net.blobs['mbox_loc'].data.shapeOut[19]: (1, 80388)

In [20]: net.blobs['mbox_conf_softmax'].data.shape
Out[20]: (1, 20097, 201)

In [21]: 80388 / 20097
Out[21]: 4
```

It is one for each score, as expected. Now, we just need to reshape it.
 

# References

- [SSD: Single Shot MultiBox Detector][1]
- [weiliu89/caffe][2]
- [weiliu89/caffe issue 208][3]


[1]: https://arxiv.org/abs/1512.02325
[2]: https://github.com/weiliu89/caffe/tree/ssd
[3]: https://github.com/weiliu89/caffe/issues/208
[4]: https://arxiv.org/abs/1405.3531

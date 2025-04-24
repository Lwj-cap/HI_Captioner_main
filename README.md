# Environment setup

Clone the repository and create the m2release conda environment using the environment.yml file:

```
conda env create -f environment.yml
conda activate Captioner
```

Then download spacy data by executing the following command:

```
python -m spacy download en
Note: Python 3.6 is required to run our code.
```



# Data preparation

To run the code, annotations and detection features for the COCO dataset are needed. Please download the annotations file annotations.zip [annotations.zip](https://ailb-web.ing.unimore.it/publicfiles/drive/meshed-memory-transformer/annotations.zip "点击下载annotations.zip") and extract it.



To reproduce our result, please download the COCO features file coco_detections.hdf5

[coco_detections.hdf5]: https://ailb-web.ing.unimore.it/publicfiles/drive/show-control-and-tell/coco_detections.hdf5

(~53.5 GB), in which detections of each image are stored under the <image_id>_features key. <image_id> is the id of each COCO image, without leading zeros (e.g. the <image_id> for COCO_val2014_000000037209.jpg is 37209), and each value should be a (N, 2048) tensor, where N is the number of detections.


### Final Project Proposal: Deepfake Detection Using Deep Learning

Ben Felter, Tibor Thompson, Travis Senf

**Problem Statement and Originality:**

As generative image models rapidly advance, creating highly realistic deepfakes has become easier, raising significant concerns about privacy, security, and misinformation. Our project focuses on developing a deep learning model to accurately distinguish real images from deepfakes and benchmark it across various deepfake sophistication levels across time. This approach is unique in its systematic evaluation of the model's effectiveness against a range of deepfake techniques, providing valuable insights into the improvement of deepfakes over time and where our ability to accurately classify them is trending.

**Proposed Methodology**

We plan to approach this problem through two primary strategies:

1. Developing a Custom Model Inspired by YoloV8: Leveraging the strengths of YoloV8's architecture, known for its efficiency in object detection, we will design a model from scratch tailored to the nuances of deepfake detection. This approach allows for the exploration of innovative feature extraction and classification techniques specifically optimized for identifying the subtle artifacts characteristic of deepfakes.

2. Fine-tuning Existing Pre-trained Models: We may also try fine-tuning one of several renowned models, including VGG-16, ResNet50, Inceptionv3, and EfficientNet, on deepfake detection tasks. These models have demonstrated exceptional performance in various image classification challenges and offer a robust foundation for identifying deepfake images. The fine-tuning process will involve adjusting the models to specialize in deepfake characteristics, utilizing transfer learning to capitalize on their pre-trained capabilities.

**Datasets**

To train and evaluate our models, we could use the following datasets:

- Real People: CelebA, VGGFace2, and IMDB-WIKI, to provide a diverse range of genuine human images.
- Deep Fakes: Celeb-DF and FaceForensics++, offering a wide variety of deepfake sophistication levels for comprehensive testing.

**Feasibility**

We have outlined a set of milestones that include dataset preparation, model development and training, evaluation, and optimization phases. Preliminary research on various classifier models indicates that the chosen datasets and models are well-suited for our project goals, and that our project is achievable within the timeframe of a semester.

**Milestones**

1. Finalize project scope - how many levels of sophistication for deepfakes, image vs video, etc (End of week 7)
2. Model Training - for each time period, train classifier (end of week 8)
3. Evaluation/Optimization, testing (end of week 9)
4. Documentation/Presentation preparation (end of week 10)
5. Present

**Midterm Status Progress Report: (10%)**

Format: 2 page PDF, submitted on Gradescope.

Content: Summarize your achievements and any deviations from the original proposal. Showcase an intermediate result through visualization. Provide an updated timeline and objectives.

**Actual work we can do to show progress:**

Find one dataset and train a simple model on it. Then we either advance the model or incorporate more datasets. See how this goes, if that's a lot of work then that'll be good.

We start with:
- CelebA
- CelebDF

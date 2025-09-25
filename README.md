# Visual-variation-learning-and-strange-poses-of-objects
## My master's thesis project
The main goal of the project was to test how the method presented in [Learning visual variation for object recognition](https://doi.org/10.1016/j.imavis.2020.103912) paper will handle data specifically prepared to mislead current classifiers.

During the project, in some of the experiments the method was slightly modified and the auxiliary task was changed from classification to regression.

---

`resnet_50_variation_finetuning.py` is used to finetune, pretrained on ImageNet dataset, modified model (the model with auxiliary task of classification) on iLab-2M model.

---

`finetuning_on_ax.py` is used to finetune, pretrained on ImageNet dataset, model on dataset containing adversarial examples.
<br>`finetuning_on_ax_with_var.py` is used to finetune, pretrained on ImageNet dataset, modified model (the model with auxiliary task of regression) on dataset containing adversarial examples.
<br>`finetuning_on_ilab_with_var.py` is used to finetune, pretrained on ImageNet dataset, modified model (the model with auxiliary task of regression) on iLab-2M model.

---

`evaluate_on_ax.py` is used to evaluate trained model on dataset containing adversarial examples.
<br>`evaluate_on_ax_with_var.py` is used to evaluate trained and modified model (the model with auxiliary task of regression) on dataset containing adversarial examples.
<br>`evaluate_on_ilab_with_var.py` is used to evaluate trained and modified model (the model with auxiliary task of regression) on iLab-2M dataset.
<br>`evaluate_plain_resnet_on_ax.py` is used to evaluate pretrained ResNet-50 model on dataset containing adversarial examples.

---

`training_logs_monitoring.py` is used to analyze the logs and charts created during all the trainings.

---
It is needed to install all the dependencies to run the scripts. 
In this case, they were launched using Python 3.12.3 in WSL (Ubuntu). The requirements are stored in `requirements.txt` file.

## Dataset used for training the models:
- [iLab-2M](https://bmobear.github.io/assets/pdf/iLab2M.pdf)
- [Custom datasets and 3D models sources](https://drive.google.com/drive/folders/1AQ8Ah7PeXPko_jcfMnH8_oJqKmiSjevt?usp=sharing) 
(Custom datasets were created using framework shared by the authors of paper: [Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects](https://doi.org/10.48550/arXiv.1811.11553). The original framework is available at this [link](https://github.com/airalcorn2/strike-with-a-pose). The fork of the repository with scripts used to create these custom datasets is available at this [link](https://github.com/bartSzczepaniec/strike-with-a-pose-modified).)

In the forked repository the following files were used to create datasets:

`/paper_code/generate_dataset.py` was used to create custom_dataset_v1 and custom_dataset_v2
`/paper_code/generate_smaller_dataset_from_v2.py` was used to create custom_dataset_v3 from custom_dataset_v2



## References
- The method tested in this project was used to improve object classification CNNs by adding auxiliary task regarding pose classification was presented in the research paper: [Learning visual variation for object recognition](https://doi.org/10.1016/j.imavis.2020.103912)

- Custom Datasets were created using framework shared by the authors of paper: [Strike (with) a Pose: Neural Networks Are Easily Fooled by Strange Poses of Familiar Objects](https://doi.org/10.48550/arXiv.1811.11553)


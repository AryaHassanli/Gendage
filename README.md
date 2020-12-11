<!-- [![Contributors][contributors-shield]][contributors-url] -->
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/AryaHassanli/Gendage">
    <img src="https://arya.li/gendage/facial-recognition.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Gendage</h3>

  <p align="center">
    Gender Recognition and Age Estimation
    <br />
    <!--<a href="https://github.com/AryaHassanli/Gendage"><strong>Explore the docs »</strong></a>
    <br />
    <br /> -->
    <a href="https://github.com/AryaHassanli/Gendage">View Demo</a>
    ·
    <a href="https://github.com/AryaHassanli/Gendage/issues">Report Bug</a>
    ·
    <a href="https://github.com/AryaHassanli/Gendage/issues">Request Feature</a>
  </p>
</p>

<details open="open">
<summary><h2 style="display: inline-block">Table of Contents</h2></summary>

- [About The Project](#about-the-project)
  * [Built With](#built-with)
- [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Downloading Datasets](#downloading-datasets)
    + [AgeDB](#agedb)
    + [UTKFace](#utkface)
    + [How to use AgeDB and UTKFace](#how-to-use-agedb-and-utkface)
- [Structure](#structure)
- [Usage](#usage)
  * [Estimate Age and Gender for Images and Videos](#estimate-age-and-gender-for-images-and-videos)
    + [Example](#example)
  * [Online - Webcam](#online---webcam)
  * [Train/Test a Classifier](#train-test-a-classifier)
  * [Using Config File](#using-config-file)
  * [Add a Classifier](#add-a-classifier)
  * [Add an Encoder](#add-an-encoder)
  * [Add a Dataset](#add-a-dataset)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [References](#references)


</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[Gendage](https://github.com/AryaHassanli/Gendage) is part of [SPRING](https://spring-h2020.eu/) project to estimate the age and the gender of multiple persons in a scene using Deep Neural Networks.

### Built With

* [Pytorch](https://pytorch.org/)
* [Facenet](https://github.com/davidsandberg/facenet)
* [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/)
* [UTKFace](https://susanqq.github.io/UTKFace/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

To begin with, you have to prepare some prerequisites.

1. Python 3.8 or greater
  
    install python using the instruction provided on [their website](https://www.python.org/downloads/)

2. `pip`

    * If you are using **Windows**:

        Most probably you have the pip by default, but to be sure that you have the latest version, let's install `pip`:

        ```sh
        py -m pip install --upgrade pip
        ```

        To check that it is installed correctly using

        ```sh
        py -m pip --version
        ```

        If it is installed correctly you will have something like this as output:

        ```sh
        pip 20.2.4 from C:\Python38\lib\site-packages\pip (python 3.8)
        ```

    * On **Mac** or **Linux**:

        `pip` is included in most of the distributions. You can also install the latest version using:

        ```sh
        python3 -m pip install --user --upgrade pip
        ```

        To test if you have installed `pip` correctly, check the version using:

        ```sh
        $ python3 -m pip --version
        pip 9.0.1 from $HOME/.local/lib/python3.6/site-packages (python 3.6)
        ```

    _For further information or help refer to [`pip` installation guide](https://pip.pypa.io/en/stable/installing/)._

### Installation

0. Change directory to base directory that you want to install the project.

1. Clone the repo

   ```sh
   git clone https://github.com/AryaHassanli/Gendage.git
   ```

2. Install required packages

   ```sh
   pip install -r requirements.txt
   ```

### Downloading Datasets

In order to train a network, you have to download the desired datasets. AgeDB and UTKFace DataLoaders are included in the source code. These are the datasets that were used to train the available pretrained models. You can use them to evaluate the pre-trained models or train new models. However, the dataset itself is not available on the repository.

#### AgeDB

AgeDB consists of more than 16K images. The images are not aligned and cropped. You can use the pre-process function or alignment option while training that will be described later.

Access to AgeDB is only possible by asking the author. The instruction is given on their [webpage](https://ibug.doc.ic.ac.uk/resources/agedb/).

#### UTKFace

UTKFace comes with an aligned version. It includes over 20K in the wild images. To download UTKFace refer to their [website](https://susanqq.github.io/UTKFace/). You can find the unaligned and aligned version. It is possible to use the aligned version without any further pre-process or alignment.

#### How to use AgeDB and UTKFace

After downloading the compressed files, move them to the datasets folder. The code itself decompress them when you run the code for the first time. For the AgeDB, the compressed file is password protected. You can either decompress it using the password you received from the author or put the password at the very begining line of `dataLoaders/AgeDB.py` by change the `zip_pass=b'UNKWONW'` to `zip_pass=b'PASSWORD'` where `PASSWORD` is the one that the AgeDB author has provided.

Rather than using the provided `datasets` folder it is possibe to create a folder with any name anywhere on your disk. In this case, the path to the new datasets folder should be set in CLI or config file.

The datasets folder structure would be as below:

!!TODO!!

## Structure

The program repository consist of a main program file named `gendage.py` which is the CLI handler, and five folders that are described below:
* `config` folder holds the config files. Each config file is a json file that includes all or some arguments to make the cli commands shorter and easier.
* `dataLoaders` includes [dataset handler](#) for each dataset.

!!TODO!!

<!-- USAGE EXAMPLES -->
## Usage

### Estimate Age and Gender for Images and Videos

```sh
usage: gendage.py run [-h] [--config_file NAME] [--input_file PATH] [--output_dir PATH] [--encoder ENC] [--encoder_pretrain MODEL] [--features FEATURE [FEATURE ...]]
                      [--classifiers NET [NET ...]] [--classifier_pretrain MODEL [MODEL ...]] [--num_classes N [N ...]]

optional arguments:
  -h, --help            show this help message and exit
  --config_file NAME    Config file name. e.g. train_config if the config file is train_config.py (default: None)
  --input_file PATH     PATH to input file. (default: None)
  --output_dir PATH     PATH to save the outputs. e.g. /artifacts/output/ or output (default: output)
  --encoder ENC         The Encoder network.choices: mobilenet_v3_small (default: mobilenet_v3_small)
  --encoder_pretrain MODEL
                        The Encoder pretrained model.choices: models/encoder/mobilenet_v3_small_1.pt (default: models/encoder/mobilenet_v3_small_1.pt)
  --features FEATURE [FEATURE ...]
                        Features to train or test.choices: age, gender (default: ['gender', 'age'])
  --classifiers NET [NET ...]
                        Network for each learning task.choices: simple (default: ['simple', 'simple'])
  --classifier_pretrain MODEL [MODEL ...]
                        The classifier pretrained model.choices: models/classifier/age_model.pt (default: ['models/classifier/gender_model.pt', 'models/classifier/age_model.pt'])
  --num_classes N [N ...]
                        Number of classes for each task. e.g. 120 2 (default: [2, 120])
```

_Accepted formats:_
* _Images: .jpg .jpeg .png_
* _Videos: Will be available soon_

#### Example

* Estimate age and gender of faces in friends.jpg:
  
  ```sh
  $ gendage run --input_file friends.jpg
  
  mobilenet_v3_small Encoder is Found!
  simple Classifier is Found!
  simple Classifier is Found!

  Detected: A 32.84 years old male
  Detected: A 26.23 years old female
  Detected: A 26.54 years old male
  Detected: A 31.46 years old female
  Detected: A 30.48 years old female
  Detected: A 29.4 years old female
  Detected: A 43.06 years old male
  Output saved on: home/user/Gendage/output/labeled_friends.jpg
  ```

### Online - Webcam

!!TODO!!

### Train/Test a Classifier

```sh
usage: gendage.py train_classifier [-h] [--config_file NAME] [--datasets_dir PATH] [--output_dir PATH] [--encoder ENC] [--encoder_pretrain MODEL] [--features FEATURE [FEATURE ...]]
                                   [--datasets DS [DS ...]] [--classifiers NET [NET ...]] [--classifier_pretrain MODEL [MODEL ...]] [--num_classes N [N ...]] [--preload PRE]
                                   [--split_size SIZE SIZE SIZE] [--batch_size BATCH] [--epochs EPOCHS] [--lr LR]

optional arguments:
  -h, --help            show this help message and exit
  --config_file NAME    Config file name. e.g. train_config if the config file is train_config.py (default: None)
  --datasets_dir PATH   PATH to datasets directory. e.g. /home/datasets/ or datasets (default: datasets)
  --output_dir PATH     PATH to save the outputs. e.g. /artifacts/output/ or output (default: output)
  --encoder ENC         The Encoder network.choices: mobilenet_v3_small (default: mobilenet_v3_small)
  --encoder_pretrain MODEL
                        The Encoder pretrained model.choices: models/encoder/mobilenet_v3_small_1.pt (default: models/encoder/mobilenet_v3_small_1.pt)
  --features FEATURE [FEATURE ...]
                        Features to train or test.choices: age, gender (default: ['gender', 'age'])
  --datasets DS [DS ...]
                        List of datasets for each task.choices: UTKFace, AgeDB (default: ['AgeDB', 'UTKFace'])
  --classifiers NET [NET ...]
                        Network for each learning task.choices: simple (default: ['simple', 'simple'])
  --classifier_pretrain MODEL [MODEL ...]
                        The classifier pretrained model.choices: models/classifier/age_model.pt (default: ['None', 'None'])
  --num_classes N [N ...]
                        Number of classes for each task. e.g. 120 2 (default: [2, 120])
  --preload PRE         Set to 1 to load the whole dataset to memory at beginningchoices: 0, 1 (default: 0)
  --split_size SIZE SIZE SIZE
                        Specify the train, validation and test size. e.g. 0.7 0.2 0.1 (default: [0.7, 0.2, 0.1])
  --batch_size BATCH    Batch size. e.g. 128 (default: 128)
  --epochs EPOCHS       Number of epochs. e.g. 40 (default: 30)
  --lr LR               Learning Rate. e.g. 0.001 (default: 0.002)
```

!!TODO!!

### Using Config File

Instead of providing arguments using CLI, it is possible to create a json file and use it as list of arguments. **All** arguments except `config_file` are requierd to run the program. If an argument is missing in config file or CLI, the default value will be used. The default value for each argument is available by `gendage run -h`. 
Therefore, the default set of arguments (mentioned in `gendage run -h`) will be used by the program by default! If `--config_file` is given, the arguments of config file will replace the default arguments. Then, the arguments given by CLI will replace both. A sample of config file for each main command is available in `config/`.

### Add a Classifier

!!TODO!!

### Add an Encoder

!!TODO!!

### Add a Dataset

!!TODO!!

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/AryaHassanli/Gendage/issues) for a list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Seyyed Arya Hassanli <br/>
[a_hassanli [at] outlook.com](mailto:a_hassanli@outlook.com) <br/>
[@A_Hassanli](https://twitter.com/A_Hassanli) <br/>
[Linkedin](https://www.linkedin.com/in/seyyed-arya-hassanli/)

Project Link: [https://github.com/AryaHassanli/Gendage](https://github.com/AryaHassanli/Gendage)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* The Gendage has been done under the supervision of:
  * Prof. Elisa Ricci: <small>[Personal Website](http://elisaricci.eu/), [Google Scholar](https://scholar.google.ca/citations?user=xf1T870AAAAJ&hl=en) </small>
  * Prof. Wei Wang: <small>[GitHub](https://weiwangtrento.github.io/), [Google Scholar](https://scholar.google.it/citations?hl=en&user=k4SdlbcAAAAJ) </small>
  * Mr. Levi Osterno Vasconcelos: <small>[Linkedin](https://www.linkedin.com/in/leviovasconcelos/) </small>
* and with the collaboration of Alessandro Conti ([GitHub](https://github.com/altndrr))

<small>
- Icons made by <a href="https://www.flaticon.com/authors/smashicons" title="Smashicons">Smashicons</a> from <a href="https://www.flaticon.com/" title="Flaticon"> www.flaticon.com</a>
<br/>
- Table of contents generated with <a href='http://ecotrust-canada.github.io/markdown-toc/'>markdown-toc</a>
</small>

## References

1. Moschoglou, S., Papaioannou, A., Sagonas, C., Deng, J., Kotsia, I., & Zafeiriou, S. (2017). [AgeDB: The First Manually Collected, In-The-Wild Age Database](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Moschoglou_AgeDB_The_First_CVPR_2017_paper.pdf). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.
2. Zhang, & Qi, H. (2017). [Age Progression/Regression by Conditional Adversarial Autoencoder](https://arxiv.org/abs/1702.08423). In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
3. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/AryaHassanli/Gendage/contributors
[forks-shield]: https://img.shields.io/github/forks/AryaHassanli/Gendage.svg?style=for-the-badge
[forks-url]: https://github.com/AryaHassanli/Gendage/network/members
[stars-shield]: https://img.shields.io/github/stars/AryaHassanli/Gendage.svg?style=for-the-badge
[stars-url]: https://github.com/AryaHassanli/Gendage/stargazers
[issues-shield]: https://img.shields.io/github/issues/AryaHassanli/Gendage.svg?style=for-the-badge
[issues-url]: https://github.com/AryaHassanli/Gendage/issues
[license-shield]: https://img.shields.io/github/license/AryaHassanli/Gendage.svg?style=for-the-badge
[license-url]: https://github.com/AryaHassanli/Gendage/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/seyyed-arya-hassanli/

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![Pull Requests][pulls-shield]][pulls-url]
[![License][license-shield]][license-url]
[![closed Pull Requests][closed_pulls-shield]][closed_pulls-url]
[![closed Issues][closed_issues-shield]][closed_issues-url]



<!-- PROJECT LOGO -->
<p align="center">
  <!-- <a href="https://github.com/TristanBandat/genre-prediction-video">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->
  <h3 align="center">Predicting the genre of an artist/a band based on their music video clip</h3>
  <p align="center">
    This project tries to predict the music genre of a given music video clip.
    <br />
    <a href="https://github.com/TristanBandat/genre-prediction-video"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/TristanBandat/genre-prediction-video">View Demo</a>
    · -->
    <a href="https://github.com/TristanBandat/genre-prediction-video/issues">Report Bug</a>
    ·
    <a href="https://github.com/TristanBandat/genre-prediction-video/issues">Request Feature</a>
  </p>
<!-- </p> -->



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <!-- <li><a href="#prerequisites">Prerequisites</a></li> -->
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <!-- <li><a href="#acknowledgements">Acknowledgements</a></li> -->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

The goal of this project is to create a 
TODO: ADD HERE


### Built With

* [PyCharm](https://www.jetbrains.com/pycharm/)
* [Vim](https://www.vim.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [TFDS CLI](https://www.tensorflow.org/datasets/cli)
* [Anaconda](https://www.anaconda.com/)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

<!-- ### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ``` -->

### Installation

1. Python Env Setup <br>
   1. Windows 
      1. Install [Anaconda](https://www.anaconda.com/)
      2. Open Anaconda Prompt and type:
         ```shell
         conda update -n base -c defaults conda
         conda create --name Python3.10 python=3.10
         conda activate Python3.10
         conda install pandas matplotlib numpy
         pip install tensorflow-datasets
         python -m pip install "tensorflow<2.11"
         ```
      3. If a GPU is available, it should be listed with the following command:
         ```shell
         python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
         ```
   2. Ubuntu
      1. Install Anaconda like stated [here](https://docs.anaconda.com/anaconda/install/linux/)
      2. Open terminal and type:
         ```shell
         conda create --name Python3.10 python=3.10
         conda activate Python3.10
         conda install pandas matplotlib numpy
         conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
         python3 -m pip install tensorflow tensorflow-datasets
         ```
      3. If a GPU is available, it should be listed with the following command:
         ```shell
         python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
         ```

2. Clone the repo
   ```shell
   git clone https://github.com/TristanBandat/genre-prediction-video.git
   ```

3. Download the data files:<br>
   _Note: The files need to be placed in a folder called `data/`._
   * [id_vgg19.tsv.bz2](https://zenodo.org/record/6609677/files/id_vgg19.tsv.bz2?download=1)
   * [id_resnet.tsv.bz2](https://zenodo.org/record/6609677/files/id_resnet.tsv.bz2?download=1)
   * [id_incp.tsv.bz2](https://zenodo.org/record/6609677/files/id_incp.tsv.bz2?download=1)
   * [id_genres_tf-idf.tsv.bz2](https://zenodo.org/record/6609677/files/id_genres_tf-idf.tsv.bz2?download=1)

4. Create dataset<br>
   ```shell
   cd datasets/Music4AllOnionDC/
   tfds build Music4AllOnionDC.py --data_dir [CWD]/data/
   ```
   


TODO: ADD HERE


   

<!-- USAGE EXAMPLES -->
## Usage

TODO: ADD HERE


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/TristanBandat/genre-prediction-video/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create.<br> 
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_Note: Look into the TODO file for open features and in which release they will be included._



<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Tristan Bandat - [@TBandat](https://twitter.com/TBandat) - tristan.bandat@gmail.com
Philipp Meingaßner - p.meingassner@gmail.com

Project Link: [https://github.com/TristanBandat/genre-prediction-video](https://github.com/TristanBandat/genre-prediction-video)



<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* []()
* []()
* []()

-->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TristanBandat/genre-prediction-video.svg?style=for-the-badge
[contributors-url]: https://github.com/TristanBandat/genre-prediction-video/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TristanBandat/genre-prediction-video.svg?style=for-the-badge
[forks-url]: https://github.com/TristanBandat/genre-prediction-video/network/members
[issues-shield]: https://img.shields.io/github/issues/TristanBandat/genre-prediction-video.svg?style=for-the-badge
[issues-url]: https://github.com/TristanBandat/genre-prediction-video/issues
[pulls-shield]: https://img.shields.io/github/issues-pr/TristanBandat/genre-prediction-video.svg?style=for-the-badge
[pulls-url]: https://github.com/TristanBandat/genre-prediction-video/pulls
[license-shield]: https://img.shields.io/github/license/TristanBandat/genre-prediction-video.svg?style=for-the-badge
[license-url]: https://github.com/TristanBandat/genre-prediction-video/blob/master/LICENSE.txt
[closed_pulls-shield]: https://img.shields.io/github/issues-pr-closed/TristanBandat/genre-prediction-video?style=for-the-badge
[closed_pulls-url]: https://github.com/TristanBandat/genre-prediction-video/pulls?q=is%3Apr+is%3Aclosed
[closed_issues-shield]: https://img.shields.io/github/issues-closed/TristanBandat/genre-prediction-video?style=for-the-badge
[closed_issues-url]: https://github.com/TristanBandat/genre-prediction-video/issues?q=is%3Aissue+is%3Aclosed

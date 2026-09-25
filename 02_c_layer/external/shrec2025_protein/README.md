## SHREC 2025: Protein Shape Classification

This repository contains the NIT Tsuyama College team's codebase for
the [SHREC 2025: Protein Shape Classification track](http://shrec2025.drugdesign.fr/).

### Environment

We run the codebase on Ubuntu 24.04.2 LTS in WSL2 of Windows 11.
The libraries and tools used are as follows:

- Point Cloud Library (PCL) 1.14.0
- Eigen 3.4.0
- CMake 3.28.3
- Python 3.13.2
- scikit-learn 1.6.1

We install PCL, Eigen and CMake using the apt-get command:

```bash
$ sudo apt-get install libpcl-dev libeigen3-dev cmake
```

Furthermore, we use [mise](https://github.com/jdx/mise) to install Python
and [uv](https://github.com/astral-sh/uv) to install scikit-learn.

### Usage

Build the codebase and prepare the dataset as follows:

```bash
$ cd ~
$ git clone https://github.com/yoshoku/shrec2025_protein.git
$ cd shrec2025_protein/bin
$ cmake ../src
$ make
$ cd ../dataset
$ wget https://shrec2025.drugdesign.fr/files/train_set.csv
$ wget https://shrec2025.drugdesign.fr/files/test_set.csv
$ cd train_set
$ wget https://shrec2025.drugdesign.fr/files/train_set.tar.xz
$ tar Jxvf train_set.tar.xz
$ cd ../test_set
$ wget https://shrec2025.drugdesign.fr/files/test_set.tar.xz
$ tar Jxvf test_set.tar.xz
```

Execute the following command to generate the feature files from the VTK files and the submission file:

```bash
$ cd ~/shrec2025_protein/dataset/train_set
$ find . -name "*.vtk" | sed -e s/\.vtk// | awk '{print $1 ".vtk " $1 ".dat"}' | xargs -t -n 2 ../../bin/vtk2feat.bin
$ cd ../test_set
$ find . -name "*.vtk" | sed -e s/\.vtk// | awk '{print $1 ".vtk " $1 ".dat"}' | xargs -t -n 2 ../../bin/vtk2feat.bin
$ cd ../../ml
$ uv sync
$ uv run main.py > submission.csv
```

### Evaluation

The evaluation script and SHREC2025 results are available in the following repository:

https://gitlab.com/ycbtaher/shrec2025

### License

This project is licensed under the [MIT License](https://github.com/yoshoku/shrec2025_protein/blob/main/LICENSE.txt).
In addition, the codebase uses the [sobol sequence generator](https://people.sc.fsu.edu/~jburkardt/cpp_src/sobol/sobol.html) that is implemented by John Burkardt.
It is licensed under the MIT Licsense.

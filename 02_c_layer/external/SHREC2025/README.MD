# SHREC2025
SHREC 2025: Protein Shape Classification
https://shrec2025.drugdesign.fr/

## Hardware requirement
This repository has been tested on Linux Ubuntu (24.04 LTS) with 64-bit Intel CPUs. Some binaries are not compatible with other CPU architectures (e.g., Apple Silicon Mac).

## Environment Setup
We recommend using conda to create a clean and reproducible environment.

### 1. Create Environment from environment.yml
```bash
conda env create -f environment.yml
conda activate shrec2025
pip3 install -r requirements.txt
```
This will install all necessary packages specified in the YAML file.


### 2. Prepare data
The data can be downloaded by following steps.

```bash
curl -o data/train_set.tar.xz https://shrec2025.drugdesign.fr/files/train_set.tar.xz
curl -o data/train_set.csv https://shrec2025.drugdesign.fr/files/train_set.csv
curl -o data/test_set.tar.xz https://shrec2025.drugdesign.fr/files/test_set.tar.xz
curl -o data/test_set.csv https://shrec2025.drugdesign.fr/files/test_set.csv

mkdir -p data/train_set
tar -xJf data/train_set.tar.xz -C data/train_set
mkdir -p data/test_set
tar -xJf data/test_set.tar.xz -C data/test_set
```

## Run the method

### 1. calculate 3DZD
```bash
# for all train_set, do
mkdir -p output/train_set/
python3 vtk2zd.py data/train_set/8ugd_8:R:3U_model1.vtk output/train_set/8ugd_8:R:3U_model1
#
# for all test_set, do
mkdir -p output/test_set/
python3 vtk2zd.py data/test_set/2320.vtk output/test_set/2320

```
To process the whole dataset, you can do with [GNU Parallel](http://www.gnu.org/software/parallel/) as following:
```bash
export OMP_NUM_THREADS=1
# for train_set
mkdir -p output/train_set/
for file in $(ls data/train_set/|cut -f 1 -d "."); do
    echo "python3 vtk2zd.py data/train_set/${file}.vtk output/train_set/${file}";
done | parallel -j 100% --bar

# for test_set
mkdir -p output/test_set/
for file in $(ls data/test_set/|cut -f 1 -d "."); do
    echo "python3 vtk2zd.py data/test_set/${file}.vtk output/test_set/${file}";
done | parallel -j 100% --bar
```

### 2. calculate volume
```bash
python3 calc_volume.py data/train_set.csv output/train_set/ > output/train_set_volume.csv
python3 calc_volume.py data/test_set.csv output/test_set/ > output/test_set_volume.csv
```

### 3. run notebook
You can now run `make_predictions.ipynb`. It will generate `test_predictions.csv` and `test_class_distance_matrix.csv`.


## Citation:
[Taher Yacoub, Camille Depenveiller, ..., Yuki Kagaya, Joon Hong Park, Daisuke Kihara, ..., Matthieu Montes (28 Authors). SHREC 2025: Protein surface shape retrieval including electrostatic potential. Computers & Graphics 132, 104394, doi:10.1016/j.cag.2025.104394 (2025).](https://www.sciencedirect.com/science/article/pii/S0097849325002353)
```
@article{YACOUB2025104394,
    title = {SHREC 2025: Protein surface shape retrieval including electrostatic potential},
    journal = {Computers & Graphics},
    volume = {132},
    pages = {104394},
    year = {2025},
    doi = {https://doi.org/10.1016/j.cag.2025.104394},
    url = {https://www.sciencedirect.com/science/article/pii/S0097849325002353},
    author = {Taher Yacoub and Camille Depenveiller and Atsushi Tatsuma and Tin Barisin and Eugen Rusakov and Udo Göbel and Yuxu Peng and Shiqiang Deng and Yuki Kagaya and Joon Hong Park and Daisuke Kihara and Marco Guerra and Giorgio Palmieri and Andrea Ranieri and Ulderico Fugacci and Silvia Biasotti and Ruiwen He and Halim Benhabiles and Adnane Cabani and Karim Hammoudi and Haotian Li and Hao Huang and Chunyan Li and Alireza Tehrani and Fanwang Meng and Farnaz Heidar-Zadeh and Tuan-Anh Yang and Matthieu Montes}
}
```
# Emostyle Wrapper

Wrapper API for generating facial expression images as synthetic data.

## Getting Started

If you are on a Linux system and have `git` and `python3` (tested on Python 3.12) installed, you can follow the following steps to set up the project.

Clone the project with its submodules.
```sh
git clone --recursive https://github.com/ClementDrn/emostyle-wrapper
cd emostyle-wrapper
```

Install the Nvidia CUDA toolkit `nvcc`.
```sh
sudo apt install nvidia-cuda-toolkit
```

Install the `gcc` version matching your CUDA version (e.g. CUDA 11.2 supports g++-10 max).
Once you get your CUDA version with the `nvcc --version` command, refer to this [StakOverflow question](https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version) for more information about the supported gcc version, and install it.
```sh
# Check your CUDA version
nvcc --version
# For example, if you have CUDA 12.0, you can install g++-12.1:
sudo apt install g++-12.1
```
Note that you might need to use the `update-alternatives` package if you have multiple versions of `gcc` installed.

Assuming that you have Python 3 installed, create a virtual environment and activate it:
```sh
python3 -m venv .venv
source .venv/bin/activate
```
Note that to deactivate the virtual environment if you are done, you can run the `deactivate` command.

Then, install the required Python packages:
```sh
python3 -m pip install -r requirements.txt
```

To use the API, you need to download the pretrained models first (~1.7 GB). Running the `setup.py` script will handle this for you.
```sh
python3 setup.py
```
You should now have, in the `pretrained` directory, the weight files for these models:
- `emo_mapping_wplus_2.pt` EmoStyle by Azari B. et al. ([drive.google.com](https://drive.google.com/file/d/17C1-ACpPbFnaRNVYpDPrzNTYbFJPL_7h/view?usp=sharing))
- `emonet_8.pth` by Toisoul A. et al. ([github.com](https://github.com/face-analysis/emonet/blob/master/pretrained/emonet_8.pth))
- `resnet50_ft_weight.pkl` ResNet50 by He K. et al., trained on VggFace2 by Huang Z. et al. ([drive.google.com](https://drive.google.com/file/d/1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU/view?usp=sharing))
- `ffhq.pkl` StyleGAN2-ADA trained on FFHQ by Karras T. et al. ([nvlabs-fi-cdn.nvidia.com](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl))
- `shape_predictor_68_face_landmarks.dat.bz2` from dlib ([dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2))
- `model_ir_se50.pth` IR-SE50 model, an [InsightFace](https://github.com/deepinsight/insightface) reimplementation, trained by [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) ([drive.google.com](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing))

From this point, generating images should work.
As an example, the following command let you generate 5 random images with the emotion "happy":
```sh
python3 generate.py -n 5 -e Happy
python3 generate.py -n 5 -e Happy -c    # Use CPU instead of GPU
```


## Usage

The `generate.py` script allows you to generate images with specific facial expressions based on predefined emotions or custom valence-arousal pairs.

### Command Line Options

Below are the available options for the `generate.py` script:

| Option                                                                                                        | Description                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-h, --help`                                                                                                  | Show the help message and exit.                                                                                                                                                                                                                                                                                                                                               |
| `-n NUM_MULT, --num_mult NUM_MULT`                                                                            | Number of images multiplier (default: 1). The actual number of images to generate is equal to `num_mult * [len(emotions)\|len(valences_arousals)/2] * len(strengths)`.                                                                                                                                                                                                        |
| `-o OUTPUT_DIR, --output_dir OUTPUT_DIR`                                                                      | Directory to save generated images (default: `output`).                                                                                                                                                                                                                                                                                                                       |
| `-e EMOTIONS [EMOTIONS ...], --emotions EMOTIONS [EMOTIONS ...]`                                              | List of emotions to apply (default: `random`). Available emotions: `Happy`, `Delighted`, `Excited`, `Astonished`, `Aroused`, `Tense`, `Alarmed`, `Angry`, `Afraid`, `Annoyed`, `Distressed`, `Frustrated`, `Miserable`, `Sad`, `Gloomy`, `Depressed`, `Bored`, `Droopy`, `Tired`, `Sleepy`, `Calm`, `Relaxed`, `Satisfied`, `AtEase`, `Content`, `Serene`, `Glad`, `Pleased`. |
| `-v VALENCES_AROUSALS [VALENCES_AROUSALS ...], --valences_arousals VALENCES_AROUSALS [VALENCES_AROUSALS ...]` | List of valence-arousal pairs (e.g., `valence0 arousal0 valence1 arousal1 ...`) to apply (default: `None`). Overrides `--emotions` if specified.                                                                                                                                                                                                                              |
| `-s STRENGTHS [STRENGTHS ...], --strengths STRENGTHS [STRENGTHS ...]`                                         | List of strengths (from `0.0` to `1.0`) for each emotion (default: `1.0` for all emotions). Requires `--emotions`. Undefined with `--valences_arousals`.                                                                                                                                                                                                                      |
| `-r, --repeat`                                                                                                | Each input image is repeatedly used until it has been edited with every emotion and strength combination.                                                                                                                                                                                                                                                                     |
| `-c, --force_cpu`                                                                                             | Force the script to run on the CPU, even if a GPU is available.                                                                                                                                                                                                                                                                                                               |

### Examples

To generate a single image with a random emotion:
```sh
python3 generate.py
```

To generate 512 images with the emotions "Happy" and "Sad" (2 emotions * 256 = 512):
```sh
python3 generate.py -n 256 -e Happy Sad
```

To generate 16 images with random emotions using the CPU:
```sh
python3 generate.py -n 16 -c
```

To generate 32 images with `(0.5, 0.7)` and `(-0.9, -0.1)` valence-arousal pairs:
```sh
python3 generate.py -n 16 -v 0.5 0.7 -0.9 -0.1
```

To generate 20 images with varying strengths for the emotions:
```sh
python3 generate.py -n 5 -e Happy Sad -s 0.5 1.0
```

To generate many facial expressions with the same input image:
```sh
python3 generate.py -n 1 -e Angry Gloomy Sad Happy -r
```


## Credits

This project is strongly based on [Emostyle](https://https://github.com/bihamta/emostyle) by Azari B. et al., which allows to generate facial expressions based on the valence-arousal model.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
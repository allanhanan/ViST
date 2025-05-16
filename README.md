# ViST  
**Vision Transformer from Scratch using only the C++ Standard Library**

ViST is a C++ implementation of the Vision Transformer (ViT) model for image classification only using the C++ standard library. Currently it classifies images into 3 classes

## Requirements
- C++17 or newer.
- Clang (or configure CMakeLists.txt)
- CMake (for building the project).
- stb_image.h for loading images

## Setup & Installation

### 1. Clone the repository  
Clone the repository to your local machine:

```bash
git clone https://github.com/allanhanan/ViST.git
cd ViST
```

### 2. Build with CMake

Ensure you have **CMake** installed.

From the project root directory, create a build directory and compile:

```bash
mkdir build
cd build
cmake ..
make
```

This will generate the executable \`ViT\` in the \`build\` folder.

---

## Usage

### 1. Training the model  
To train the model, run thecommand from the project's root directory:

```bash
./ViT
```

It will start training the model using the images located in the following directory structure:

```
program_root/
└── train/
    ├── apple/
    │   ├── image1.png
    │   ├── image2.png
    │   └── image3.png
    ├── orange/
    │   ├── image1.png
    │   ├── image2.png
    │   └── image3.png
    └── banana/
        ├── image1.png
        ├── image2.png
        └── image3.png
```

**Note**: The image directory path is currently hardcoded in the source code.

### 2. Testing the model  
After training the model, test it on an image by running:

```bash
./ViT /path/to/model_checkpoint.bin /path/to/test_image.png
```

Example:

```bash
./ViT /home/allan/project/viT/vit/build/model_checkpoint.bin /home/allan/project/viT/vit/test.png
```

Parameters are hardcoded for now and only supports CPU training

also uses stb_image.h so it technically isnt only using the std library but dont care + L + Ratio

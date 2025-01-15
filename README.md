# Interactive4D: Interactive 4D LiDAR Segmentation

<div align="center">
<a href="https://github.com/Ilya-Fradlin/">Ilya Fradlin</a>, 
<a href="https://www.vision.rwth-aachen.de/person/245/">Idil Esen Zulfikar</a>, 
<a href="https://github.com/YilmazKadir/">Kadir Yilmaz</a>, 
<a href="https://theodorakontogianni.github.io/"> Theodora Kontogianni </a>, 
<a href="https://www.vision.rwth-aachen.de/person/1/">Bastian Leibe</a>

RWTH Aachen University, ETH AI Center

Interactive 4D segmentation is a new paradigm that segments multiple objects across consecutive LiDAR scans in a single step, improving efficiency and consistency while simplifying tracking and annotation. Interactive4D model supports interactive 4D multi-object segmentation, where a user collaborates with a deep learning model to segment multiple 3D objects simultaneously across multiple scans, by providing interactive clicks.

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg"></a>

![teaser](./docs/Interactive4d_teaser.gif)

</div>
<br><br>

[[Project Webpage](https://vision.rwth-aachen.de/Interactive4D)] [[arXiv](https://arxiv.org/abs/2410.08206)]

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li>
      <a href="#news-newspaper">News</a>
    </li>
    <li>
      <a href="#installation-hammer">Installation</a>
    </li>
    <li>
      <a href="#data-preprocessing-triangular_ruler">Data Preprocessing</a>
    </li>
    <li>
      <a href="#training-and-evaluation-chart_with_downwards_trend">Training and Evaluation</a>
    </li>
    <li>
      <a href="#interactive-tool-computer">Interactive Tool</a>
    </li>
    <li>
      <a href="#bibtex-scroll">BibTeX</a>
    </li>
    <li>
      <a href="#acknowledgment-pray">Acknowledgment</a>
    </li>

  </ol>
  </ol>
</details>

---

## News :newspaper:

- [27/01/2025]: Interactive4D was accepted to ICRA 2025.
- [28/01/2025]: Code release.

---

## Installation :hammer:

The main dependencies of the project are the following:

```yaml
Python: 3.7
CUDA: 11.6
```

You can set up a conda environment as follows:

#### Step 1: Create an environment

```shell
git clone https://github.com/Ilya-Fradlin/Interactive4D.git
cd Interactive4D
```

```shell
conda create --name interactive4d python=3.7 pip=22.2*
conda activate interactive4d
```

#### Step 2: Install PyTorch

```shell
# adjust your CUDA version accordingly!
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Step 3: Install Minkowski

**3.1 Prepare for installation:**

```shell
conda install openblas-devel -c anaconda
# adjust your CUDA path accordingly!
export CUDA_HOME=/usr/local/cuda
 # adjust to your corresponding C++ compiler
export CXX=g++-10
```

**3.2 Installation:**

```shell
 # run the installation command
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```

If you run into issues, please also refer to Minkowski's official instructions.

#### Step 4: Install additional required packages

```shell
pip install -r requirements.txt --no-deps
```

---

## Data preprocessing :triangular_ruler:

**Dataset Preprocessing Overview**

After setting up your environment, you can preprocess datasets for both training and validation. Provide the path to your raw dataset, and the preprocessing script will generate a corresponding JSON file for the corresponding dataset, collecting information for each scan (including scene_id, scan_id, pcd_filepath, label_filepath, pose). This JSON is critical for the subsequent training or evaluation runs.

**To preprocess SemanticKITTI simply run:**

```shell
python -m datasets.preprocessing.semantickitti_preprocessing preprocess \
--data_dir "PATH_TO_RAW_SEMANTICKITTI_DATASET" \
--save_dir "datasets/jsons/"
```

**To preprocess nuScenes adjust:**

```shell
python -m datasets.preprocessing.nuscenes_preprocessing preprocess \
--data_dir "PATH_TO_RAW_NUSCENES_DATASET" \
--save_dir "datasets/jsons/"
```

**Preprocessing KITTI360 requires a bit more preparation:**

1. Single-scan Annotations

- Use the [recoverKITTI360label](https://github.com/JulesSanchez/recoverKITTI360label) script to produce single-scan annotations.
- In order to obtain **instance** labels in addition to **semantic** labels, modify _accumulation.py_ within [recoverKITTI360label](https://github.com/JulesSanchez/recoverKITTI360label) as follows (in every relevant occurrence):

```python
superpcd_static[:,[0,1,2,6]] -> superpcd_static[:,[0,1,2,7]]
superpcd_dynamic[:,[0,1,2,6]] -> superpcd_dynamic[:,[0,1,2,7]]
```

2. Once you have the per-point labels for each single scan, run:

```shell
python -m datasets.preprocessing.kitti360_preprocessing preprocess \
--data_dir "PATH_TO_KITTI360_DATASET" \
--label_dir "PATH_TO_KITTI360_SINGLE_SCAN_LABELS" \  # The recovered labels
--save_dir "datasets/jsons/"

```

---

## Training and Evaluation :chart_with_downwards_trend:

<details open 
  style="
    padding: 10px; 
    border: 1px solid #ccc; 
    border-radius: 8px; 
    background-color: #f6f8fa;
  "
>
  <summary><span style="font-weight: bold;"><b>Configuration Setup</b></span></summary>
  <p>
    Before running any training or evaluation jobs, ensure that the 
    <a href="conf/config.yaml">config.yaml</a> file is properly adjusted. 
    Below are the key sections that you may need to modify:
  </p>

  <!-- GENERAL -->
  <details style="
    margin-left: 1.5em; 
    margin-top: 10px; 
    border-radius: 6px;
  ">
    <summary><strong>General</strong></summary>
    <ul>
      <li><strong>ckpt_path / weights</strong>
        <ul>
          <li>Set the path to the checkpoint weights.</li>
          <li>During <strong>training</strong>, this can be used to resume from a previous checkpoint.</li>
          <li>During <strong>validation</strong>, this should point to the final or desired checkpoint.</li>
        </ul>
      </li>
      <li><strong>max_num_clicks</strong>
        <ul>
          <li>The total click budget allowed per object per scene.</li>
        </ul>
      </li>
      <li><strong>max_clicks_per_obj</strong>
        <ul>
          <li>A higher limit preventing excessive clicks on a single object.</li>
          <li>Helps avoid “click waste” during unproductive refinements.</li>
        </ul>
      </li>
      <li><strong>mode</strong>
        <ul>
          <li>Specifies whether to run in <strong>train</strong> or <strong>validate</strong> mode.</li>
          <li>Each mode references a dedicated section in the config (e.g., <code>modes.train</code> or <code>modes.validate</code>).</li>
        </ul>
      </li>
    </ul>
  </details>

  <!-- DATA -->
  <details style="
    margin-left: 1.5em; 
    margin-top: 10px; 
    border-radius: 6px;
  ">
    <summary><strong>Data</strong></summary>
    <ul>
      <li><strong>dataset</strong>
        <ul>
          <li>Which dataset to use (requires a preprocessed JSON).</li>
          <li><em>Note</em>: Training is currently only supported on SemanticKITTI, but adjustments can be made for other sets (e.g., nuScenes, KITTI360).</li>
        </ul>
      </li>
      <li><strong>window_overlap</strong>
        <ul>
          <li>The number of scans overlapping between consecutive temporal windows.</li>
        </ul>
      </li>
      <li><strong>sweep</strong>
        <ul>
          <li>The number of LiDAR scans concatenated for each sample.</li>
        </ul>
      </li>
      <li><strong>data_dir</strong>
        <ul>
          <li>The directory containing the preprocessed JSON files.</li>
        </ul>
      </li>
    </ul>
  </details>

  <!-- CLICKING STRATEGY -->
  <details style="
    margin-left: 1.5em; 
    margin-top: 10px; 
    border-radius: 6px;
  ">
    <summary><strong>Clicking Strategy</strong></summary>
    <ul>
      <li><strong>rank_error_strategy</strong>
        <ul>
          <li>Dictates which error region to target each time the model needs an additional click (options: SI, BD).</li>
        </ul>
      </li>
      <li><strong>initial_clicking_strategy</strong>
        <ul>
          <li>Specifies where to click within the error region upon the first encounter (options: centroid, random, boundary_dependent, dbscan).</li>
        </ul>
      </li>
      <li><strong>refinement_clicking_strategy</strong>
        <ul>
          <li>Defines where to click in subsequent encounters for refining the segmentation (same options as initial).</li>
        </ul>
      </li>
    </ul>
  </details>

  <!-- LOGGING -->
  <details style="
    margin-left: 1.5em; 
    margin-top: 10px; 
    border-radius: 6px;
  ">
    <summary><strong>Logging</strong></summary>
    <ul>
      <li><strong>WandB Integration</strong>
        <ul>
          <li>Adjust <code>project_name</code>, <code>workspace</code>, and <code>entity</code> to match your Weights & Biases setup.</li>
          <li>An API key may be required if running on a remote server or cluster.</li>
        </ul>
      </li>
      <li><strong>visualization_frequency</strong>
        <ul>
          <li>Defines how often to log visualizations to WandB (e.g., point clouds, ground truth).</li>
        </ul>
      </li>
      <li><strong>save_predictions</strong>
        <ul>
          <li>If set to true, saves predictions locally to the specified <code>save_dir</code>.</li>
        </ul>
      </li>
    </ul>
  </details>
</details>

---

### Training :rocket:

<details open 
  style="
    padding: 10px; 
    border: 1px solid #ccc; 
    border-radius: 8px; 
    background-color: #f6f8fa;
  "
>
  <summary><span style="font-weight: bold;"><b>Training Details</b></span></summary>
  <ul>
    <li><strong>Dataset Support</strong>
      <ul>
        <li>Training is currently supported on SemanticKITTI. (It can be adapted to nuScenes, etc., but modification in the data loader would be required.)</li>
      </ul>
    </li>
    <li><strong>Multi-GPU Setup</strong>
      <ul>
        <li>To train our model, we use 16×NVIDIA-A40 GPUs, each with 40GB memory.</li>
        <li>If the training is running in a SLURM cluster, ensure that the number of nodes and GPUs match both your SLURM command and the <code>trainer</code> settings in <code>config.yaml</code>, e.g.:</li>
        <li>
          <pre><code>sbatch --nodes=4 --ntasks-per-node=4 --gres=gpu:4 \
       --output=outputs/%j_Interactive4d.txt \
       scripts/job_submissions/run_on_node.sh
          </code></pre>
        </li>
      </ul>
    </li>
    <li><strong>Batch Size &amp; Learning Rate</strong>
      <ul>
        <li>Default batch size is 1 to handle memory constraints.</li>
        <li>When running multi-GPU training, however, the effective batch size is the number of GPUs multiplied by the local batch size. i.e. 16 in the discussed case above</li>
        <li>The learning rate is automatically scaled according to the number of GPUs.</li>
      </ul>
    </li>
  </ul>
</details>

**Run the training script:**

```shell
./scripts/train.sh
```

---

### Evaluation :chart_with_upwards_trend:

After training the model / using the the provided weights you can download here:

- Interactive4D - 3D setup (sweep=1) - [here](https://omnomnom.vision.rwth-aachen.de/data/interactive4d/interactive4d.ckpt)
- Interactive4D - 4D setup (sweep=4) - [here](https://omnomnom.vision.rwth-aachen.de/data/interactive4d/interactive4d_sweep4.ckpt)

<details open 
  style="
    padding: 10px; 
    border: 1px solid #ccc; 
    border-radius: 8px; 
    background-color: #f6f8fa;
  "
>
  <summary><span style="font-weight: bold;"><b>Evaluation Details</b></span></summary>
  <ul>
    <li><strong>Checkpoint Weights</strong>
      <ul>
        <li>Update <code>ckpt_path</code> in <code>config.yaml</code> to point to the desired checkpoint if you are evaluating a different set of weights.</li>
      </ul>
    </li>
    <li><strong>Single-GPU Setup</strong>
      <ul>
        <li>Typically uses a 3090 GPU with 24GB of memory.</li>
      </ul>
    </li>
    <li><strong>Logging &amp; Visualization</strong>
      <ul>
        <li>The <code>visualization_frequency</code> controls how frequently point clouds are uploaded to WandB (e.g., every <em>n</em> steps).</li>
        <li>Excessive logging may slow down the evaluation.</li>
      </ul>
    </li>
    <li><strong>Saving Predictions</strong>
      <ul>
        <li>If <code>save_predictions</code> is set to <code>true</code>, predictions will be saved to the directory specified in <code>prediction_dir</code>.</li>
        <li>These can be used for further analysis, e.g., calculating panoptic quality.</li>
      </ul>
    </li>
    <li><strong>Multi-Sweep Models</strong>
      <ul>
        <li>Multi-sweep setups (e.g., <code>sweep=4</code> vs. <code>sweep=10</code>) require compatible weights.</li>
        <li>Ensure your training and evaluation sweeps match, unless you specifically want to test generalization to a different number of sweeps.</li>
      </ul>
    </li>
  </ul>
</details>

**Run the evaluation script:**

```shell
./scripts/evaluate.sh
```

---

## Interactive Tool :computer:

The interactive tool is a user-friendly interface based on the [AGILE3D Indoor annotator](https://github.com/ywyue/AGILE3D), enhanced to handle large and sparse outdoor environments effectively. This tool simplifies the process of segmenting LiDAR data by enabling real-time interaction with the model, making it intuitive for both researchers and practitioners.

**User Guide:**
A comprehensive guide to using the tool, including setup and interaction steps, is available in the [Interactive Tool Documentation](interactive_tool/segmentation_tool.md).

---

## BibTeX :scroll:

```text
@article      {fradlin2024interactive4d,
  title     = {{Interactive4D: Interactive 4D LiDAR Segmentation}},
  author    = {Fradlin, Ilya and Zulfikar, Idil Esen and Yilmaz, Kadir and Kontogianni, Theodora and Leibe, Bastian},
  journal   = {arXiv preprint arXiv:2410.08206},
  year      = {2024}
}
```

---

## Acknowledgment :pray:

**We sincerely thank all the volunteers who participated in our user study!**
The computing resources for most of the experiments were granted by the Gauss Centre for Supercomputing e.V. through the John von Neumann Institute for Computing on the GCS Supercomputer JUWELS at Julich Supercomputing Centre.
Theodora Kontogianni is a postdoctoral research fellow at the ETH AI Center and her research is partially funded by the Hasler Stiftung Grant project (23069). Idil Esen Zulfikar’s research is funded by the BMBF project NeuroSys-D (03ZU1106DA). Kadir Yilmaz's research is funded by the Bosch-RWTH LHC project Context Understanding for Autonomous Systems.

Portions of our code are built upon the foundations of [Mask4Former](https://github.com/YilmazKadir/Mask4Former) and [AGILE3D](https://github.com/ywyue/AGILE3D).

# Dataset Sources and Selection

## Primary Dataset: Chest X-ray Pneumothorax

Preferred source:

- Kaggle dataset: `vbookshelf/pneumothorax-chest-xray-images-and-masks`
- Reason: PNG images and PNG masks are already paired and easier to use for a fast classification-plus-explanation benchmark.
- Size noted on Kaggle: about 24.1k files and 4.89 GB.
- Use case: primary quantitative validation of explanation maps against pneumothorax masks.

Alternative source:

- Kaggle dataset: `jesperdramsch/siim-acr-pneumothorax-segmentation-data`
- Reason: closer to the original SIIM-ACR competition data, with chest X-rays and dense RLE annotations.
- Tradeoff: requires more preprocessing because masks are RLE encoded.

Competition/context:

- SIIM-ACR Pneumothorax Segmentation Kaggle Challenge.
- Task: classify and segment pneumothorax on chest radiographs.
- The SIIM page states that 1,475 teams participated and 352 submitted results during evaluation.

Recommended first command, after Kaggle credentials are configured:

```bash
mkdir -p .kaggle data_local/cxr_pneumothorax
# Put kaggle.json in .kaggle/ and do not commit it.
PATH="$HOME/.local/bin:$PATH" KAGGLE_CONFIG_DIR="$PWD/.kaggle" kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks -p data_local/cxr_pneumothorax --unzip
```

If the Kaggle CLI is already on `PATH`, this shorter command is equivalent:

```bash
KAGGLE_CONFIG_DIR="$PWD/.kaggle" kaggle datasets download -d vbookshelf/pneumothorax-chest-xray-images-and-masks -p data_local/cxr_pneumothorax --unzip
```

## Secondary Dataset: Head CT Intracranial Hemorrhage

Option A: Public CT dataset with masks

- Kaggle dataset: `vbookshelf/computed-tomography-ct-images`
- Reason: includes head CT images and associated intracranial hemorrhage masks for a subset.
- Kaggle summary: 2,500 brain-window images, 2,500 bone-window images, 82 patients, and 318 images with associated intracranial masks.
- Original source cited by Kaggle: PhysioNet CT ICH dataset by Hssayeni.
- Use case: low-risk CT pilot with existing masks.

Recommended first command:

```bash
mkdir -p .kaggle data_local/head_ct_ich
PATH="$HOME/.local/bin:$PATH" KAGGLE_CONFIG_DIR="$PWD/.kaggle" kaggle datasets download -d vbookshelf/computed-tomography-ct-images -p data_local/head_ct_ich --unzip
```

Option B: RSNA Intracranial Hemorrhage Detection

- Kaggle competition: `rsna-intracranial-hemorrhage-detection`
- Reason: large and clinically important head CT classification benchmark.
- RSNA states the challenge dataset contains more than 25,000 annotated cranial CT exams.
- A 2023 open-access article reports 752,799 DICOM slices from 18,938 patients and labels for hemorrhage presence/subtypes.
- Tradeoff: no dense lesion masks in the main competition labels, so manual masks are needed for explanation-localization validation.

Recommended first command, if competition access is configured:

```bash
mkdir -p .kaggle data_local/rsna_ich
PATH="$HOME/.local/bin:$PATH" KAGGLE_CONFIG_DIR="$PWD/.kaggle" kaggle competitions download -c rsna-intracranial-hemorrhage-detection -p data_local/rsna_ich
```

## Selection Rule

Use the easiest path that gives lesion masks quickly:

1. Primary: `vbookshelf/pneumothorax-chest-xray-images-and-masks`.
2. CT pilot: `vbookshelf/computed-tomography-ct-images` if masks are adequate.
3. RSNA IHD or local CT only if a small manual mask subset can be created without delaying the thesis.

## Data Safety

Do not commit downloaded images, masks, DICOM files, or derived patient images to git.
